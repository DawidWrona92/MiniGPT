#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minigpt.py v2 — edukacyjny, minimalny GPT od zera (poziom znakowy), jeden plik.

Komendy:
    prepare   teksty -> tokeny + słownik znaków
    train     trening (CPU lub GPU CUDA/ROCm), wznawianie, akumulacja gradientu, log CSV
    sample    generowanie (KV-cache, temperatura, top-k, top-p, kary za powtórzenia)
    inspect   rozkład następnego znaku, attention per warstwa i głowa, logit lens
    eval      loss walidacyjny, loss dla zadanego tekstu, ablacja głów
    bench     pomiar tok/s, pamięci i szybkości generowania (bez danych)
    selftest  testy poprawności modelu na losowych wagach (CPU, kilka sekund)

Przykładowy przebieg (E0 -> E1):
    python3 minigpt.py selftest
    python3 minigpt.py bench
    python3 minigpt.py prepare --inputs "data/*.txt"
    python3 minigpt.py train --name base --steps 3000 --dropout 0.2
    python3 minigpt.py sample --name base --which best --prompt "Król "
    python3 minigpt.py inspect --name base --prompt "Wawel znajduje się w " --per_head --lens

Zależności: torch, numpy (opcjonalnie matplotlib do PNG z attention)

Attention jest liczone ręcznie (QK^T, maska, softmax, @V), żeby każdą operację dało się
obejrzeć. Ścieżka szybka (F.scaled_dot_product_attention) działa tylko w treningu, gdy nie
potrzeba macierzy attention, i daje ten sam wynik (sprawdza to selftest).
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

TOKEN_DTYPE = np.uint16
UNKNOWN_CODE = 0xFFFF
CHUNK_CHARS = 4 * 1024 * 1024


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """ Self-attention z maską przyczynową (token widzi tylko siebie i lewą stronę).

    Ręcznie: Q,K,V -> score = Q @ K^T / sqrt(d_head) -> maska -> softmax -> @ V.
    Obsługuje KV-cache (past) oraz bramkę głów (head_gate) do ablacji.
    """

    def __init__(self, d_model, n_head, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model musi być podzielne przez n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout
        self.fast = True
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, past=None, head_gate=None, want_attn=False):
        batch_size, seq_len, channels = x.shape
        query, key, value = self.qkv(x).split(channels, dim=2)             # po (B, T, C)
        # (B, T, C) -> (B, n_head, T, d_head): każda głowa dostaje swój wycinek kanałów
        query = query.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        if past is not None:
            key = torch.cat([past[0], key], dim=2)                         # klucze z cache + nowe
            value = torch.cat([past[1], value], dim=2)
        present = (key, value)
        total_len = key.size(2)

        use_fast = self.fast and not want_attn and head_gate is None and past is None
        if use_fast:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            attn = None
        else:
            # score: jak bardzo i-ty token słucha j-tego; 1/sqrt(d_head) trzyma softmax w zakresie
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_head)      # (B, H, T, T_total)
            # maska przyczynowa: token i nie patrzy na j>i; -inf daje po softmaxie dokładnie 0
            causal = self.mask[total_len - seq_len:total_len, :total_len]
            scores = scores.masked_fill(~causal, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = self.attn_dropout(attn) @ value                          # (B, H, T, d_head)

        if head_gate is not None:
            out = out * head_gate.to(out.dtype).view(1, -1, 1, 1)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)   # sklej głowy
        return self.resid_dropout(self.proj(out)), present, attn


class Block(nn.Module):
    """ Blok transformera pre-norm: x + attn(LN(x)), potem x + mlp(LN(x)). """

    def __init__(self, d_model, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        # MLP 4x: d_model -> 4*d_model -> d_model (indeks 2 to projekcja wyjściowa, ważna przy inicjalizacji)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, past=None, head_gate=None, want_attn=False):
        attn_out, present, attn = self.attn(self.ln1(x), past, head_gate, want_attn)
        x = x + attn_out                                                   # residual
        x = x + self.mlp(self.ln2(x))
        return x, present, attn


def _run_block(block, x):
    return block(x)[0]


class MiniGPT(nn.Module):
    """ Decoder-only GPT: embedding znaku + embedding pozycji -> stos bloków -> LN -> head.

    Head jest związany z embeddingiem (tied): ta sama macierz koduje znak na wejściu
    i odczytuje logity na wyjściu.
    """

    def __init__(self, vocab_size, block_size, d_model, n_layer, n_head, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.use_checkpoint = False
        self.config = dict(vocab_size=vocab_size, block_size=block_size, d_model=d_model,
                           n_layer=n_layer, n_head=n_head, dropout=dropout)
        self.wte = nn.Embedding(vocab_size, d_model)                       # co to za znak
        self.wpe = nn.Embedding(block_size, d_model)                       # na której pozycji stoi
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_head, block_size, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init_weights)
        # projekcje wracające do strumienia residual: mniejsze odchylenie, bo sumuje się 2*n_layer wkładów
        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_fast_attention(self, enabled):
        for block in self.blocks:
            block.attn.fast = enabled

    def forward(self, idx, targets=None, past=None, head_gate=None, use_cache=False, collect=False):
        """ Zwraca (logits, loss); z use_cache lub collect dodatkowo słownik info:
        info["past"] (cache K,V), info["attn"] (attention per warstwa), info["hidden"] (strumień residual:
        indeks 0 to embedding, indeks i+1 to wyjście bloku i) """
        batch_size, seq_len = idx.shape
        past_len = 0 if past is None else past[0][0].size(2)
        assert past_len + seq_len <= self.block_size, "sekwencja dłuższa niż block_size"
        positions = torch.arange(past_len, past_len + seq_len, device=idx.device)
        x = self.embed_dropout(self.wte(idx) + self.wpe(positions))        # (B, T, C)

        presents, attns, hiddens = [], [], [x]
        checkpointing = (self.use_checkpoint and self.training and not use_cache
                         and not collect and head_gate is None and past is None)
        for layer_index, block in enumerate(self.blocks):
            if checkpointing:
                x = checkpoint(_run_block, block, x, use_reentrant=False)
                continue
            layer_past = None if past is None else past[layer_index]
            layer_gate = None if head_gate is None else head_gate[layer_index]
            x, present, attn = block(x, layer_past, layer_gate, collect)
            if use_cache:
                presents.append(present)
            if collect:
                attns.append(attn)
                hiddens.append(x)

        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)                              # tied head: (B, T, vocab)
        loss = None
        if targets is not None:
            # cross-entropy: jak bardzo model mylił się co do następnego znaku
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        if use_cache or collect:
            info = {}
            if use_cache:
                info["past"] = presents
            if collect:
                info["attn"] = attns
                info["hidden"] = hiddens
            return logits, loss, info
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repeat_penalty=1.0, repeat_window=64, no_repeat_ngram=0, use_cache=True):
        """ Autoregresja: przewiduj znak, dopisz, powtórz.

        Z cache okno pozycji jest absolutne, więc gdy cache zapełni block_size, kontekst jest
        przeliczany od nowa z ostatnich block_size/2 znaków (bez cache okno przesuwa się o 1). """
        was_training = self.training
        self.eval()
        blocker = NgramBlocker(idx, no_repeat_ngram) if no_repeat_ngram > 1 else None
        past = None
        for _ in range(max_new_tokens):
            if use_cache:
                if past is not None and past[0][0].size(2) >= self.block_size:
                    past = None
                    context = idx[:, -(self.block_size // 2):]
                elif past is None:
                    context = idx[:, -self.block_size:]
                else:
                    context = idx[:, -1:]
                logits, _, info = self(context, past=past, use_cache=True)
                past = info["past"]
            else:
                logits, _ = self(idx[:, -self.block_size:])
            next_token = sample_next_token(logits[:, -1, :], idx, temperature, top_k, top_p,
                                           repeat_penalty, repeat_window, blocker)
            idx = torch.cat([idx, next_token], dim=1)
            if blocker is not None:
                blocker.append(next_token)
        if was_training:
            self.train()
        return idx

    def num_params(self):
        return sum(param.numel() for param in self.parameters())


# ----------------------------------------------------------------------------
# PRÓBKOWANIE
# ----------------------------------------------------------------------------

class NgramBlocker:
    """ Zakazuje znaku, który domknąłby n-gram już widziany w historii (per wiersz batcha). """

    def __init__(self, idx, ngram_size):
        self.prefix_len = ngram_size - 1
        self.histories = [row.tolist() for row in idx]
        self.seen = [dict() for _ in self.histories]
        for row_index, history in enumerate(self.histories):
            for start in range(len(history) - self.prefix_len):
                prefix = history[start:start + self.prefix_len]
                self._register(row_index, prefix, history[start + self.prefix_len])

    def _register(self, row_index, prefix, token):
        self.seen[row_index].setdefault(tuple(prefix), set()).add(token)

    def apply(self, logits):
        for row_index, history in enumerate(self.histories):
            if len(history) < self.prefix_len:
                continue
            prefix = tuple(history[len(history) - self.prefix_len:])
            banned = self.seen[row_index].get(prefix)
            if not banned:
                continue
            saved = logits[row_index].clone()
            logits[row_index, list(banned)] = float("-inf")
            if torch.isinf(logits[row_index]).all():                       # nic nie zostało: odpuść zakaz
                logits[row_index] = saved
        return logits

    def append(self, next_token):
        for row_index, token in enumerate(next_token[:, 0].tolist()):
            history = self.histories[row_index]
            if len(history) >= self.prefix_len:
                self._register(row_index, history[len(history) - self.prefix_len:], token)
            history.append(token)


def apply_repeat_penalty(logits, idx, penalty, window):
    """ Kara w stylu CTRL: znaki z ostatnich `window` pozycji dostają słabszy logit.
    Na poziomie znakowym to narzędzie tępe (karze też spacje i samogłoski), lepiej działa no_repeat_ngram. """
    if penalty == 1.0:
        return logits
    recent = idx[:, -window:]
    picked = logits.gather(1, recent)
    picked = torch.where(picked > 0, picked / penalty, picked * penalty)
    return logits.scatter(1, recent, picked)


def filter_top_k(logits, top_k):
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k).values[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits, top_p):
    """ Nucleus: zostaw najmniejszy zbiór znaków o łącznej masie >= top_p (najlepszy zawsze zostaje). """
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = (cumulative - sorted_probs) >= top_p                   # masa przed tym znakiem już wystarcza
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_index, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(logits, idx, temperature, top_k, top_p, repeat_penalty, repeat_window, blocker):
    logits = apply_repeat_penalty(logits.clone(), idx, repeat_penalty, repeat_window)
    if blocker is not None:
        logits = blocker.apply(logits)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)                  # zachłannie
    logits = logits / temperature                                          # temperatura: ostrość rozkładu
    if top_k:
        logits = filter_top_k(logits, top_k)
    if top_p is not None:
        logits = filter_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ----------------------------------------------------------------------------
# NARZĘDZIA: urządzenie, dane, checkpointy
# ----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.version.hip is not None:
        print("UWAGA: PyTorch z ROCm, ale GPU niedostępne. Sprawdź HSA_OVERRIDE_GFX_VERSION=10.3.0 "
              "(RX 6700 XT to gfx1031), grupy render/video i restart sesji.")
    return "cpu"


def print_device_info(device):
    if device.startswith("cuda"):
        backend = f"ROCm/HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
        print(f"Urządzenie: {torch.cuda.get_device_name(0)} ({backend})")
        if torch.version.hip:
            print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'nie ustawione')}")
    else:
        print("UWAGA: trening na CPU. Wolne, ale zadziała.")


def autocast_context(device, amp_enabled):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled)


def torch_load(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def atomic_save(state, path):
    temp_path = path + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def checkpoint_path(data_dir, name, which):
    suffix = "_best" if which == "best" else ""
    return os.path.join(data_dir, f"ckpt_{name}{suffix}.pt")


def build_model(config):
    return MiniGPT(**{"dropout": 0.0, **config})


def load_model(data_dir, name, which, device):
    path = checkpoint_path(data_dir, name, which)
    if not os.path.isfile(path):
        raise SystemExit(f"Brak checkpointu: {path}")
    ckpt = torch_load(path, device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.isfile(path):
        raise SystemExit(f"Brak {path}. Uruchom najpierw: prepare")
    with open(path, encoding="utf-8") as file:
        meta = json.load(file)
    vocab = meta["vocab"]
    stoi = {char: index for index, char in enumerate(vocab)}
    return vocab, stoi


def open_tokens(data_dir, split):
    path = os.path.join(data_dir, f"tokens_{split}.bin")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def encode(text, stoi):
    unknown = sorted({char for char in text if char not in stoi})
    if unknown:
        print(f"UWAGA: pomijam znaki spoza słownika: {unknown!r}")
    return [stoi[char] for char in text if char in stoi]


def display_char(char):
    return {"\n": "⏎", " ": "␣", "\t": "⇥"}.get(char, char)


def batch_from_positions(data, starts, block_size, device):
    offsets = np.arange(block_size)
    inputs = data[starts[:, None] + offsets].astype(np.int64)
    targets = data[starts[:, None] + offsets + 1].astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def make_train_batch_fn(data, block_size, batch_size, device, seed):
    rng = np.random.default_rng(seed)

    def next_batch():
        starts = rng.integers(0, len(data) - block_size, size=batch_size)
        return batch_from_positions(data, starts, block_size, device)
    return next_batch


def make_val_positions(data_length, block_size, batch_size, iters, seed=1234):
    """ Stałe okna walidacyjne: krzywe val z różnych kroków i przebiegów są porównywalne. """
    rng = np.random.default_rng(seed)
    return rng.integers(0, data_length - block_size, size=(iters, batch_size))


@torch.no_grad()
def evaluate_loss(model, data, positions, block_size, device, amp_enabled=False, head_gate=None):
    was_training = model.training
    model.eval()
    losses = []
    for starts in positions:
        inputs, targets = batch_from_positions(data, starts, block_size, device)
        with autocast_context(device, amp_enabled):
            _, loss = model(inputs, targets, head_gate=head_gate)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def make_optimizer(model, lr, weight_decay):
    """ AdamW: decay tylko dla macierzy (>=2 wymiary), bez LayerNorm. """
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def learning_rate_at(step, total_steps, base_lr, warmup_steps, min_ratio):
    """ Liniowy warmup, potem cosine decay do min_ratio * base_lr. """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def train_step(model, optimizer, scaler, next_batch, accum_steps, device, amp_enabled, grad_clip):
    """ Jeden krok optymalizatora z akumulacją gradientu. Zwraca (średni loss, norma gradientu). """
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for _ in range(accum_steps):
        inputs, targets = next_batch()
        with autocast_context(device, amp_enabled):
            _, loss = model(inputs, targets)
        scaler.scale(loss / accum_steps).backward()
        loss_sum += loss.detach() / accum_steps
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss_sum, grad_norm


def sync_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# DANE: prepare
# ----------------------------------------------------------------------------

def iter_text_chunks(path, normalize):
    with open(path, encoding="utf-8", errors="ignore") as file:
        while True:
            chunk = file.read(CHUNK_CHARS)
            if not chunk:
                break
            yield unicodedata.normalize("NFC", chunk) if normalize else chunk


def validation_chars(length, val_fraction):
    if val_fraction <= 0:
        return 0
    return max(int(length * val_fraction), min(length // 10, 8192))


def cmd_prepare(args):
    """ Czyta pliki tekstowe, buduje słownik znaków, zapisuje tokeny jako surowe uint16.

    Walidacja to ciągły ogon każdego pliku (val_fraction), więc okna walidacyjne nie mają szwów.
    Poziom znakowy zamiast BPE jest celowy: słownik jest mały, a badasz wnętrze modelu, nie tokenizer. """
    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)))
    paths = [path for path in dict.fromkeys(paths) if os.path.isfile(path)]
    if not paths:
        raise SystemExit("Brak plików. Podaj wzorzec, np. --inputs 'data/*.txt'")
    os.makedirs(args.data_dir, exist_ok=True)
    normalize = not args.no_normalize

    counts = Counter()
    file_lengths = {}
    for path in paths:                                                     # przebieg 1: słownik i długości
        length = 0
        for chunk in iter_text_chunks(path, normalize):
            counts.update(chunk)
            length += len(chunk)
        file_lengths[path] = length
    total_chars = sum(file_lengths.values())
    print(f"Plików: {len(paths)} | znaków: {total_chars:,} | różnych znaków: {len(counts)}")

    if args.vocab_from:
        vocab, _ = load_meta(args.vocab_from)
        print(f"Słownik z {args.vocab_from}: {len(vocab)} znaków")
    else:
        vocab = sorted(char for char, count in counts.items() if count >= args.min_count)
        if "\n" not in vocab:
            vocab = sorted(vocab + ["\n"])
    if len(vocab) >= UNKNOWN_CODE:
        raise SystemExit(f"Słownik za duży ({len(vocab)}). Zwiększ --min_count.")
    stoi = {char: index for index, char in enumerate(vocab)}
    dropped_chars = sum(count for char, count in counts.items() if char not in stoi)
    print(f"Słownik: {len(vocab)} znaków | pominięte rzadkie znaki: {dropped_chars:,} "
          f"({100.0 * dropped_chars / max(1, total_chars):.3f}% tekstu)")

    lookup = np.full(0x110000, UNKNOWN_CODE, dtype=TOKEN_DTYPE)
    for char, index in stoi.items():
        lookup[ord(char)] = index

    def encode_part(text):
        code_points = np.frombuffer(text.encode("utf-32-le", errors="ignore"), dtype=np.uint32)
        ids = lookup[code_points]
        return ids[ids != UNKNOWN_CODE]

    newline_token = np.array([stoi["\n"]], dtype=TOKEN_DTYPE)
    train_tmp = os.path.join(args.data_dir, "tokens_train.bin.tmp")
    val_tmp = os.path.join(args.data_dir, "tokens_val.bin.tmp")
    train_tokens = val_tokens = 0
    with open(train_tmp, "wb") as train_file, open(val_tmp, "wb") as val_file:
        for path in paths:                                                 # przebieg 2: tokenizacja
            length = file_lengths[path]
            val_start = length - validation_chars(length, args.val_fraction)
            offset = 0
            for chunk in iter_text_chunks(path, normalize):
                split_at = min(max(val_start - offset, 0), len(chunk))
                offset += len(chunk)
                train_ids = encode_part(chunk[:split_at])
                val_ids = encode_part(chunk[split_at:])
                train_file.write(train_ids.tobytes())
                val_file.write(val_ids.tobytes())
                train_tokens += len(train_ids)
                val_tokens += len(val_ids)
            train_file.write(newline_token.tobytes())                      # separator plików
            train_tokens += 1
    os.replace(train_tmp, os.path.join(args.data_dir, "tokens_train.bin"))
    os.replace(val_tmp, os.path.join(args.data_dir, "tokens_val.bin"))
    meta = dict(vocab=vocab, normalize=normalize, min_count=args.min_count)
    with open(os.path.join(args.data_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False)
    print(f"Tokenów: train {train_tokens:,} | val {val_tokens:,}")
    print(f"Zapisano w {args.data_dir}/: meta.json, tokens_train.bin, tokens_val.bin")


# ----------------------------------------------------------------------------
# TRENING: train
# ----------------------------------------------------------------------------

CSV_FIELDS = ["step", "train_loss", "val_loss", "lr", "grad_norm", "tok_per_s", "elapsed_s"]
RESUME_KEYS = ("batch_size", "accum", "lr", "min_lr_ratio", "warmup", "weight_decay",
               "grad_clip", "amp", "checkpoint", "manual_attn")


def save_checkpoint(path, model, optimizer, scaler, step, best_val, args):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "config": model.config, "step": step,
             "best_val": best_val,
             "args": {key: value for key, value in vars(args).items() if key != "func"}}
    atomic_save(state, path)


def cmd_train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    print_device_info(device)
    vocab, stoi = load_meta(args.data_dir)
    train_data = open_tokens(args.data_dir, "train")
    if train_data is None:
        raise SystemExit("Brak tokens_train.bin. Uruchom najpierw: prepare")
    val_data = open_tokens(args.data_dir, "val")

    last_path = checkpoint_path(args.data_dir, args.name, "last")
    best_path = checkpoint_path(args.data_dir, args.name, "best")
    log_path = os.path.join(args.data_dir, f"log_{args.name}.csv")
    resume_ckpt = None
    if args.resume:
        if os.path.isfile(last_path):
            resume_ckpt = torch_load(last_path, device)
        else:
            print(f"UWAGA: --resume, ale brak {last_path}. Start od zera.")
    elif os.path.isfile(last_path) and not args.overwrite:
        raise SystemExit(f"Istnieje {last_path}. Użyj --resume, --overwrite albo innej --name.")

    if resume_ckpt is not None:
        # hiperparametry z checkpointu, chyba że podano je jawnie w wierszu poleceń
        explicit = {token[2:].split("=")[0] for token in sys.argv if token.startswith("--")}
        saved_args = resume_ckpt.get("args", {})
        for key in RESUME_KEYS:
            if key in saved_args and key not in explicit:
                setattr(args, key, saved_args[key])
        config = resume_ckpt["config"]
    elif args.init_from:
        init_ckpt = torch_load(args.init_from, device)
        config = {**init_ckpt["config"], "dropout": args.dropout}
    else:
        config = dict(vocab_size=len(vocab), block_size=args.block_size, d_model=args.d_model,
                      n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout)
    if config["vocab_size"] != len(vocab):
        raise SystemExit(f"Słownik ({len(vocab)}) nie pasuje do modelu ({config['vocab_size']}). "
                         f"Dla dotrenowania użyj prepare --vocab_from.")
    block_size = config["block_size"]
    if len(train_data) <= block_size + 1:
        raise SystemExit("Za mało danych treningowych względem block_size.")
    if val_data is not None and len(val_data) <= block_size + 1:
        print("UWAGA: zbiór walidacyjny krótszy niż block_size, pomijam walidację.")
        val_data = None

    model = build_model(config).to(device)
    model.use_checkpoint = args.checkpoint
    model.set_fast_attention(not args.manual_attn)
    start_step, best_val = 0, float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        start_step = resume_ckpt["step"]
        best_val = resume_ckpt.get("best_val", best_val)
        print(f"Wznawiam od kroku {start_step}")
        if start_step >= args.steps:
            raise SystemExit(f"Checkpoint jest już na kroku {start_step} >= --steps {args.steps}. "
                             f"Podaj większe --steps, aby kontynuować.")
    elif args.init_from:
        model.load_state_dict(init_ckpt["model"])
        print(f"Wagi startowe z {args.init_from} (krok 0, nowy optymalizator)")

    amp_enabled = args.amp and device.startswith("cuda")
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        if amp_enabled and "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

    total_steps = args.steps
    warmup_steps = max(1, min(args.warmup, total_steps // 10))
    tokens_per_step = args.batch_size * block_size * args.accum
    next_batch = make_train_batch_fn(train_data, block_size, args.batch_size, device, args.seed + start_step)
    val_positions = None
    if val_data is not None:
        val_positions = make_val_positions(len(val_data), block_size, args.batch_size, args.val_iters)

    print(f"Konfiguracja: {config}")
    print(f"Parametry: {model.num_params():,} | tokenów/krok: {tokens_per_step:,} "
          f"(batch {args.batch_size} x akumulacja {args.accum}) | "
          f"łącznie {tokens_per_step * total_steps:,} ≈ {tokens_per_step * total_steps / len(train_data):.2f} epoki")
    print(f"Loss startowy powinien wynosić około ln(vocab) = {math.log(len(vocab)):.3f}")

    if start_step == 0 and os.path.isfile(log_path):
        os.replace(log_path, f"{log_path}.{int(time.time())}.bak")
    new_log = not os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    if new_log:
        log_writer.writeheader()

    def log_row(**values):
        log_writer.writerow(values)
        log_file.flush()

    prompt_ids = encode(args.preview_prompt, stoi) or [stoi.get("\n", 0)]
    run_start = time.time()
    interval_start = time.time()
    interval_loss = torch.zeros((), device=device)
    interval_steps = 0
    last_train_loss = float("nan")
    done_steps = start_step
    model.train()
    try:
        for step in range(start_step, total_steps):
            lr = learning_rate_at(step, total_steps, args.lr, warmup_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, grad_norm = train_step(model, optimizer, scaler, next_batch, args.accum,
                                         device, amp_enabled, args.grad_clip)
            interval_loss += loss
            interval_steps += 1
            done_steps = step + 1

            if done_steps % args.log_every == 0 or done_steps == total_steps:
                last_train_loss = (interval_loss / interval_steps).item()
                elapsed = time.time() - interval_start
                tok_per_s = interval_steps * tokens_per_step / elapsed
                print(f"step {done_steps:6d}/{total_steps} | loss {last_train_loss:.4f} | lr {lr:.2e} | "
                      f"grad {grad_norm.item():.2f} | {tok_per_s:,.0f} tok/s")
                log_row(step=done_steps, train_loss=f"{last_train_loss:.5f}", lr=f"{lr:.3e}",
                        grad_norm=f"{grad_norm.item():.4f}", tok_per_s=f"{tok_per_s:.0f}",
                        elapsed_s=f"{time.time() - run_start:.1f}")
                interval_loss = torch.zeros((), device=device)
                interval_steps = 0
                interval_start = time.time()

            if val_data is not None and (done_steps % args.val_every == 0 or done_steps == total_steps):
                val_loss = evaluate_loss(model, val_data, val_positions, block_size, device, amp_enabled)
                improved = val_loss < best_val
                print(f"  >> walidacja: {val_loss:.4f}" + ("  (najlepszy)" if improved else ""))
                log_row(step=done_steps, val_loss=f"{val_loss:.5f}", elapsed_s=f"{time.time() - run_start:.1f}")
                if improved:
                    best_val = val_loss
                    atomic_save({"model": model.state_dict(), "config": model.config, "step": done_steps,
                                 "best_val": best_val}, best_path)

            if args.preview_every and done_steps % args.preview_every == 0:
                context = torch.tensor([prompt_ids], device=device)
                preview = model.generate(context, 120, temperature=0.8, top_k=10)
                text = "".join(vocab[index] for index in preview[0].tolist())
                print("  >> próbka:", text.replace("\n", " "))

            if done_steps % args.save_every == 0:
                save_checkpoint(last_path, model, optimizer, scaler, done_steps, best_val, args)
    except KeyboardInterrupt:
        print(f"\nPrzerwano w kroku {done_steps}, zapisuję checkpoint.")
        save_checkpoint(last_path, model, optimizer, scaler, done_steps, best_val, args)
        log_file.close()
        return

    save_checkpoint(last_path, model, optimizer, scaler, done_steps, best_val, args)
    log_file.close()
    best_text = f"{best_val:.4f}" if best_val < float("inf") else "brak"
    print(f"Gotowe. Krok {done_steps}, loss treningowy {last_train_loss:.4f}, "
          f"najlepsza walidacja {best_text}. Checkpoint: {last_path}")


# ----------------------------------------------------------------------------
# GENEROWANIE: sample
# ----------------------------------------------------------------------------

def cmd_sample(args):
    device = pick_device(args.device)
    if args.seed is not None:
        set_seed(args.seed)
    model, ckpt = load_model(args.data_dir, args.name, args.which, device)
    vocab, stoi = load_meta(args.data_dir)
    if len(vocab) != model.vocab_size:
        raise SystemExit("Słownik w meta.json nie pasuje do checkpointu.")
    ids = encode(args.prompt, stoi) or [stoi.get("\n", 0)]
    context = torch.tensor([ids], device=device)

    sync_device(device)
    start = time.time()
    out = model.generate(context, args.max_tokens, args.temperature, args.top_k or None, args.top_p,
                         args.repeat_penalty, args.repeat_window, args.no_repeat_ngram,
                         use_cache=not args.no_cache)
    sync_device(device)
    elapsed = time.time() - start
    print("".join(vocab[index] for index in out[0].tolist()))
    print(f"\n[{args.max_tokens / elapsed:.1f} znaków/s | cache: {'nie' if args.no_cache else 'tak'} | "
          f"krok treningu: {ckpt.get('step', '?')}]")


# ----------------------------------------------------------------------------
# INSPEKCJA: inspect
# ----------------------------------------------------------------------------

def cmd_inspect(args):
    """ Podgląd wnętrza: rozkład następnego znaku, attention per warstwa/głowa, logit lens, heatmapa. """
    device = pick_device(args.device)
    model, ckpt = load_model(args.data_dir, args.name, args.which, device)
    vocab, stoi = load_meta(args.data_dir)
    ids = encode(args.prompt, stoi)
    if len(ids) < 2:
        raise SystemExit("Podaj dłuższy prompt (min. 2 znaki w słowniku)")
    if len(ids) > model.block_size:
        print(f"UWAGA: prompt dłuższy niż block_size, biorę ostatnie {model.block_size} znaków")
        ids = ids[-model.block_size:]
    tokens = [vocab[index] for index in ids]
    seq_len = len(ids)
    inputs = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits, _, info = model(inputs, collect=True)
        final_probs = F.softmax(logits[0, -1], dim=-1)

        print(f"Prompt: {args.prompt!r}\n")
        print("Następny znak, top 10:")
        top = torch.topk(final_probs, min(10, len(vocab)))
        for prob, index in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {display_char(vocab[index]):>4}  {prob * 100:5.1f}%  {'█' * int(prob * 50)}")

        print("\nAttention ostatniego znaku (średnia po głowach), znak@pozycja(waga):")
        for layer_index, attn in enumerate(info["attn"]):
            row = attn[0].mean(dim=0)[-1].cpu()
            top_positions = sorted(torch.topk(row, min(4, seq_len)).indices.tolist())
            pairs = ", ".join(f"{display_char(tokens[pos])}@{pos}({row[pos].item():.2f})" for pos in top_positions)
            print(f"  warstwa {layer_index:2d}: {pairs}")

        if args.per_head:
            print(f"\nAttention per głowa (entropia: 0 = skupiona na jednym znaku, ln(T) = {math.log(seq_len):.2f} = równomierna):")
            for layer_index, attn in enumerate(info["attn"]):
                for head_index in range(model.n_head):
                    row = attn[0, head_index, -1].cpu()
                    entropy = -(row * torch.log(row + 1e-12)).sum().item()
                    top_positions = torch.topk(row, min(3, seq_len)).indices.tolist()
                    pairs = ", ".join(f"{display_char(tokens[pos])}@{pos}({row[pos].item():.2f})" for pos in top_positions)
                    print(f"  L{layer_index} H{head_index}  entropia {entropy:4.2f} | {pairs}")

        if args.lens:
            print("\nLogit lens (strumień residual po każdej warstwie przepuszczony przez LN i head):")
            print("  etap        top-3 znaki (prawdopodobieństwo)                      KL(końcowy || etap)")
            for stage_index, hidden in enumerate(info["hidden"]):
                label = "embedding" if stage_index == 0 else f"warstwa {stage_index - 1}"
                lens_logits = F.linear(model.ln_f(hidden[0, -1]), model.wte.weight)
                lens_probs = F.softmax(lens_logits, dim=-1)
                kl = max(0.0, (final_probs * (final_probs.clamp_min(1e-12).log()
                                              - lens_probs.clamp_min(1e-12).log())).sum().item())
                best = torch.topk(lens_probs, min(3, len(vocab)))
                cells = "  ".join(f"{display_char(vocab[index])}({prob:.2f})"
                                  for prob, index in zip(best.values.tolist(), best.indices.tolist()))
                print(f"  {label:<10}  {cells:<52}  {kl:8.3f}")

    if args.layer is None or args.head is None:
        return
    att = info["attn"][args.layer][0, args.head].cpu().float()
    print(f"\nHeatmapa attention: warstwa {args.layer}, głowa {args.head}")
    if seq_len <= 48:
        print("       " + " ".join(f"{display_char(token):>2}" for token in tokens))
        for row_index in range(seq_len):
            cells = []
            for col_index in range(seq_len):
                value = att[row_index, col_index].item()
                cells.append(f"{'·' if value < 0.02 else ('▪' if value < 0.2 else '█'):>2}")
            print(f"  {display_char(tokens[row_index]):>3}  " + " ".join(cells))
    else:
        print("  (prompt dłuższy niż 48 znaków, tekstowa heatmapa pominięta)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.imshow(att.numpy(), cmap="viridis")
        font_size = 8 if seq_len <= 64 else 4
        axis.set_xticks(range(seq_len))
        axis.set_yticks(range(seq_len))
        axis.set_xticklabels([display_char(token) for token in tokens], fontsize=font_size)
        axis.set_yticklabels([display_char(token) for token in tokens], fontsize=font_size)
        axis.set_title(f"attention L{args.layer} H{args.head}")
        out_png = os.path.join(args.data_dir, f"attn_{args.name}_L{args.layer}_H{args.head}.png")
        figure.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Zapisano też: {out_png}")
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# OCENA: eval
# ----------------------------------------------------------------------------

def parse_ablation(spec, n_layer, n_head, device):
    gate = torch.ones(n_layer, n_head, device=device)
    for item in spec.split(","):
        layer_text, head_text = item.split(":")
        gate[int(layer_text), int(head_text)] = 0.0
    return gate


def cmd_eval(args):
    """ Loss walidacyjny, loss dla tekstu (np. test prawdy przez zamianę) i ablacja głów. """
    device = pick_device(args.device)
    model, ckpt = load_model(args.data_dir, args.name, args.which, device)
    vocab, stoi = load_meta(args.data_dir)
    block_size = model.block_size

    if args.text:
        ids = encode(args.text, stoi)
        if not 2 <= len(ids) <= block_size + 1:
            raise SystemExit(f"Tekst musi mieć od 2 do {block_size + 1} znaków w słowniku.")
        inputs = torch.tensor([ids[:-1]], device=device)
        targets = torch.tensor([ids[1:]], device=device)
        with torch.no_grad():
            logits, _ = model(inputs)
        token_losses = -F.log_softmax(logits[0], dim=-1).gather(1, targets[0][:, None])[:, 0]
        total = token_losses.sum().item()
        mean = token_losses.mean().item()
        print(f"Ocenianych znaków: {len(token_losses)} | suma: {total:.3f} nat | "
              f"średnia: {mean:.4f} nat/znak ({mean / math.log(2):.3f} bitów/znak)")
        return

    val_data = open_tokens(args.data_dir, "val")
    if val_data is None or len(val_data) <= block_size + 1:
        raise SystemExit("Brak (lub za krótki) zbiór walidacyjny.")
    positions = make_val_positions(len(val_data), block_size, args.batch_size, args.iters)
    baseline = evaluate_loss(model, val_data, positions, block_size, device)
    print(f"Loss walidacyjny: {baseline:.4f} nat/znak ({baseline / math.log(2):.3f} bitów/znak) "
          f"| krok treningu: {ckpt.get('step', '?')}")

    if args.ablate:
        gate = parse_ablation(args.ablate, model.n_layer, model.n_head, device)
        loss = evaluate_loss(model, val_data, positions, block_size, device, head_gate=gate)
        print(f"Po ablacji {args.ablate}: {loss:.4f} (zmiana {loss - baseline:+.4f})")

    if args.ablate_scan:
        results = []
        for layer_index in range(model.n_layer):
            for head_index in range(model.n_head):
                gate = torch.ones(model.n_layer, model.n_head, device=device)
                gate[layer_index, head_index] = 0.0
                loss = evaluate_loss(model, val_data, positions, block_size, device, head_gate=gate)
                results.append((loss - baseline, layer_index, head_index))
        print("\nAblacja pojedynczych głów, ranking wg wzrostu lossu:")
        for delta, layer_index, head_index in sorted(results, reverse=True):
            print(f"  L{layer_index} H{head_index}  {delta:+.4f}")


# ----------------------------------------------------------------------------
# POMIAR: bench
# ----------------------------------------------------------------------------

def cmd_bench(args):
    """ Pomiar tok/s treningu i szybkości generowania na losowych danych (bez prepare). """
    set_seed(args.seed)
    device = pick_device(args.device)
    print_device_info(device)
    config = dict(vocab_size=args.vocab_size, block_size=args.block_size, d_model=args.d_model,
                  n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout)
    model = build_model(config).to(device)
    model.use_checkpoint = args.checkpoint
    model.set_fast_attention(not args.manual_attn)
    amp_enabled = args.amp and device.startswith("cuda")
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"Parametry: {model.num_params():,} | batch {args.batch_size} x akumulacja {args.accum} "
          f"x block {args.block_size} | amp: {amp_enabled} | checkpointing: {args.checkpoint} | "
          f"attention: {'ręczne' if args.manual_attn else 'szybkie (SDPA)'}")

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    def next_batch():
        return inputs, targets

    model.train()
    for _ in range(3):                                                     # rozgrzewka
        train_step(model, optimizer, scaler, next_batch, args.accum, device, amp_enabled, args.grad_clip)
    sync_device(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(args.steps):
        train_step(model, optimizer, scaler, next_batch, args.accum, device, amp_enabled, args.grad_clip)
    sync_device(device)
    elapsed = time.time() - start
    tokens = args.steps * args.batch_size * args.block_size * args.accum
    print(f"Trening: {tokens / elapsed:,.0f} tok/s ({elapsed / args.steps * 1000:.0f} ms/krok)")
    if device.startswith("cuda"):
        print(f"Szczytowa pamięć GPU: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB")

    new_tokens = min(args.gen_tokens, args.block_size - 17)
    if new_tokens < 8:
        return
    prompt = torch.randint(0, args.vocab_size, (1, 16), device=device)
    speeds = {}
    for use_cache in (False, True):
        sync_device(device)
        start = time.time()
        model.generate(prompt, new_tokens, use_cache=use_cache)
        sync_device(device)
        speeds[use_cache] = new_tokens / (time.time() - start)
        print(f"Generowanie {'z cache   ' if use_cache else 'bez cache '}: {speeds[use_cache]:.1f} znaków/s")
    print(f"Przyspieszenie dzięki KV-cache: {speeds[True] / speeds[False]:.1f}x")


# ----------------------------------------------------------------------------
# TESTY: selftest
# ----------------------------------------------------------------------------

def cmd_selftest(args):
    """ Testy poprawności na małym modelu z losowymi wagami (CPU). """
    torch.manual_seed(0)
    failures = []

    def check(name, condition, detail=""):
        print(f"  [{'OK' if condition else 'BŁĄD'}] {name}" + (f" ({detail})" if detail else ""))
        if not condition:
            failures.append(name)

    config = dict(vocab_size=20, block_size=16, d_model=32, n_layer=2, n_head=4, dropout=0.0)
    model = build_model(config).eval()
    tokens = torch.randint(0, 20, (3, 16))
    targets = torch.randint(0, 20, (3, 16))
    print("Testy modelu:")

    with torch.no_grad():
        _, loss = model(tokens, targets)
    check("loss startowy ≈ ln(vocab)", abs(loss.item() - math.log(20)) < 0.1,
          f"{loss.item():.3f} vs {math.log(20):.3f}")

    with torch.no_grad():
        logits_a, _ = model(tokens)
        changed = tokens.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 20
        logits_b, _ = model(changed)
    check("przyczynowość: zmiana ostatniego znaku nie rusza wcześniejszych logitów",
          torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6))

    with torch.no_grad():
        _, _, info = model(tokens, collect=True)
    attn = info["attn"][0]
    check("attention: wiersze sumują się do 1", torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5))
    check("attention: zero nad przekątną", float(torch.triu(attn, diagonal=1).abs().max()) == 0.0)

    with torch.no_grad():
        model.set_fast_attention(True)
        fast_logits, _ = model(tokens)
        model.set_fast_attention(False)
        manual_logits, _ = model(tokens)
        model.set_fast_attention(True)
    check("ścieżka SDPA == attention ręczne", torch.allclose(fast_logits, manual_logits, atol=1e-5),
          f"max różnica {float((fast_logits - manual_logits).abs().max()):.2e}")

    with torch.no_grad():
        full_logits, _ = model(tokens[:1])
        _, _, info = model(tokens[:1, :5], use_cache=True)
        pieces = []
        past = info["past"]
        for position in range(5, 16):
            step_logits, _, info = model(tokens[:1, position:position + 1], past=past, use_cache=True)
            past = info["past"]
            pieces.append(step_logits)
        cached_tail = torch.cat(pieces, dim=1)
    check("KV-cache: logity krok po kroku == pełny przebieg", torch.allclose(cached_tail, full_logits[:, 5:], atol=1e-5),
          f"max różnica {float((cached_tail - full_logits[:, 5:]).abs().max()):.2e}")

    prompt = tokens[:1, :4]
    with_cache = model.generate(prompt, 10, temperature=0, use_cache=True)
    without_cache = model.generate(prompt, 10, temperature=0, use_cache=False)
    check("generowanie zachłanne: z cache == bez cache", torch.equal(with_cache, without_cache))

    long_run = model.generate(prompt, 40, temperature=1.0, use_cache=True)
    check("generowanie dłuższe niż block_size działa (przeliczenie cache)", long_run.shape[1] == 44)

    with torch.no_grad():
        ones_gate = torch.ones(2, 4)
        zero_gate = torch.ones(2, 4)
        zero_gate[0] = 0.0
        gated_ones, _ = model(tokens, head_gate=ones_gate)
        gated_zero, _ = model(tokens, head_gate=zero_gate)
    check("head_gate = 1 nic nie zmienia", torch.allclose(gated_ones, fast_logits, atol=1e-5))
    check("head_gate = 0 dla całej warstwy zmienia wynik", not torch.allclose(gated_zero, fast_logits, atol=1e-5))

    plain = build_model(config).train()
    checkpointed = build_model(config).train()
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.use_checkpoint = True
    _, loss_plain = plain(tokens, targets)
    loss_plain.backward()
    _, loss_checkpointed = checkpointed(tokens, targets)
    loss_checkpointed.backward()
    grads_match = all(torch.allclose(a.grad, b.grad, atol=1e-6)
                      for a, b in zip(plain.parameters(), checkpointed.parameters()))
    check("gradient checkpointing daje te same gradienty", grads_match)

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    check("top-k zostawia k znaków", int(torch.isfinite(filter_top_k(logits, 2)).sum()) == 2)
    check("top-p zawsze zostawia najlepszy znak", int(torch.isfinite(filter_top_p(logits, 0.01)).sum()) == 1)
    check("top-p = 1 nic nie obcina", int(torch.isfinite(filter_top_p(logits, 1.0)).sum()) == 5)

    blocked = model.generate(tokens[:1, :4], 40, temperature=1.0, no_repeat_ngram=3)[0].tolist()
    trigrams = [tuple(blocked[start:start + 3]) for start in range(len(blocked) - 2)]
    check("no_repeat_ngram=3: brak powtórzonych trigramów", len(trigrams) == len(set(trigrams)))

    trainee = build_model(config).train()
    optimizer = make_optimizer(trainee, 3e-3, 0.0)
    pattern = torch.arange(20).repeat(4)
    rng = np.random.default_rng(0)
    first_loss = last_loss = 0.0
    for step in range(80):
        starts = rng.integers(0, 40, size=8)
        batch_inputs = torch.stack([pattern[start:start + 16] for start in starts])
        batch_targets = torch.stack([pattern[start + 1:start + 17] for start in starts])
        _, loss = trainee(batch_inputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    check("trening uczy się prostego wzorca (loss spada o ponad połowę)", last_loss < 0.5 * first_loss,
          f"{first_loss:.3f} -> {last_loss:.3f}")

    if failures:
        raise SystemExit(f"\nNiezaliczone testy: {len(failures)}")
    print("\nWszystkie testy zaliczone.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def add_common_args(parser, seed_default):
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--name", default="base", help="nazwa przebiegu: ckpt_<name>.pt, log_<name>.csv")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--seed", type=int, default=seed_default)


def add_shape_args(parser):
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0, help="dla małych korpusów (E1) 0.1-0.2")


def add_training_args(parser):
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum", type=int, default=1, help="akumulacja gradientu (efektywny batch = batch_size x accum)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="fp16 autocast (tylko GPU)")
    parser.add_argument("--checkpoint", action="store_true", help="gradient checkpointing: mniej pamięci, wolniej")
    parser.add_argument("--manual_attn", action="store_true", help="wyłącz ścieżkę SDPA w treningu")


def add_sampling_args(parser):
    parser.add_argument("--which", choices=["last", "best"], default="last")


def main():
    parser = argparse.ArgumentParser(description="MiniGPT PL v2 — edukacyjny GPT od zera", allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare_parser = sub.add_parser("prepare", help="teksty -> tokeny")
    prepare_parser.add_argument("--inputs", nargs="+", required=True, help="wzorce plików, np. 'data/*.txt'")
    prepare_parser.add_argument("--data_dir", default="data")
    prepare_parser.add_argument("--val_fraction", type=float, default=0.01, help="ogon każdego pliku na walidację")
    prepare_parser.add_argument("--min_count", type=int, default=1, help="znaki rzadsze są pomijane (dla Wikipedii np. 20)")
    prepare_parser.add_argument("--vocab_from", help="katalog z meta.json do użycia gotowego słownika (dotrenowanie)")
    prepare_parser.add_argument("--no_normalize", action="store_true", help="bez normalizacji Unicode NFC")
    prepare_parser.set_defaults(func=cmd_prepare)

    train_parser = sub.add_parser("train", help="trening")
    add_common_args(train_parser, seed_default=1337)
    add_shape_args(train_parser)
    add_training_args(train_parser)
    train_parser.add_argument("--steps", type=int, default=5000, help="docelowa łączna liczba kroków")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true", help="nadpisz istniejący checkpoint o tej nazwie")
    train_parser.add_argument("--init_from", help="checkpoint z wagami startowymi (dotrenowanie, nowy optymalizator)")
    train_parser.add_argument("--log_every", type=int, default=50)
    train_parser.add_argument("--val_every", type=int, default=500)
    train_parser.add_argument("--val_iters", type=int, default=20)
    train_parser.add_argument("--save_every", type=int, default=500)
    train_parser.add_argument("--preview_every", type=int, default=500)
    train_parser.add_argument("--preview_prompt", default="Król ")
    train_parser.set_defaults(func=cmd_train)

    sample_parser = sub.add_parser("sample", help="generowanie")
    add_common_args(sample_parser, seed_default=None)
    add_sampling_args(sample_parser)
    sample_parser.add_argument("--prompt", default="Król ")
    sample_parser.add_argument("--max_tokens", type=int, default=400)
    sample_parser.add_argument("--temperature", type=float, default=0.8, help="0 = zachłannie")
    sample_parser.add_argument("--top_k", type=int, default=40, help="0 = wyłączone")
    sample_parser.add_argument("--top_p", type=float, default=None)
    sample_parser.add_argument("--repeat_penalty", type=float, default=1.0, help="np. 1.1; na poziomie znakowym tępe")
    sample_parser.add_argument("--repeat_window", type=int, default=64)
    sample_parser.add_argument("--no_repeat_ngram", type=int, default=0, help="np. 8: zakaz powtórzenia ośmiogramu")
    sample_parser.add_argument("--no_cache", action="store_true", help="bez KV-cache (porównanie szybkości)")
    sample_parser.set_defaults(func=cmd_sample)

    inspect_parser = sub.add_parser("inspect", help="podgląd attention, rozkładów i logit lens")
    add_common_args(inspect_parser, seed_default=None)
    add_sampling_args(inspect_parser)
    inspect_parser.add_argument("--prompt", default="Wawel znajduje się w ")
    inspect_parser.add_argument("--layer", type=int, default=None)
    inspect_parser.add_argument("--head", type=int, default=None)
    inspect_parser.add_argument("--per_head", action="store_true", help="attention i entropia każdej głowy")
    inspect_parser.add_argument("--lens", action="store_true", help="logit lens po każdej warstwie")
    inspect_parser.set_defaults(func=cmd_inspect)

    eval_parser = sub.add_parser("eval", help="loss walidacyjny, loss tekstu, ablacja głów")
    add_common_args(eval_parser, seed_default=None)
    add_sampling_args(eval_parser)
    eval_parser.add_argument("--text", help="oceń zadany tekst zamiast zbioru walidacyjnego")
    eval_parser.add_argument("--batch_size", type=int, default=16)
    eval_parser.add_argument("--iters", type=int, default=10)
    eval_parser.add_argument("--ablate", help="wyzeruj głowy, np. '2:3,0:1' (warstwa:głowa)")
    eval_parser.add_argument("--ablate_scan", action="store_true", help="ranking wpływu każdej głowy")
    eval_parser.set_defaults(func=cmd_eval)

    bench_parser = sub.add_parser("bench", help="pomiar tok/s i szybkości generowania")
    add_common_args(bench_parser, seed_default=1337)
    add_shape_args(bench_parser)
    add_training_args(bench_parser)
    bench_parser.add_argument("--vocab_size", type=int, default=100)
    bench_parser.add_argument("--steps", type=int, default=30, help="liczba mierzonych kroków")
    bench_parser.add_argument("--gen_tokens", type=int, default=200)
    bench_parser.set_defaults(func=cmd_bench)

    selftest_parser = sub.add_parser("selftest", help="testy poprawności modelu")
    selftest_parser.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
