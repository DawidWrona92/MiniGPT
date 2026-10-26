# MiniGPT PL

Edukacyjny, minimalny GPT od zera (poziom znakowy, PyTorch), jeden plik: `minigpt.py`.
Attention liczone ręcznie (QK^T, maska przyczynowa, softmax, @V), żeby każdy krok dało się obejrzeć.

## Komendy

| Komenda | Zadanie |
| --- | --- |
| `selftest` | testy poprawności modelu na losowych wagach (CPU, kilka sekund) |
| `bench` | tok/s treningu, pamięć GPU, szybkość generowania z KV-cache i bez |
| `prepare` | teksty -> tokeny (`.bin`) + słownik znaków (`meta.json`) |
| `train` | trening: wznawianie, akumulacja gradientu, checkpointing, log CSV |
| `sample` | generowanie: temperatura, top-k, top-p, kary za powtórzenia |
| `inspect` | rozkład następnego znaku, attention per warstwa i głowa, logit lens, heatmapa |
| `eval` | loss walidacyjny, loss zadanego tekstu, ablacja głów |

## Szybki start

    python3 minigpt.py selftest
    python3 minigpt.py bench
    python3 minigpt.py prepare --inputs "data/*.txt"
    python3 minigpt.py train --name base --steps 3000 --dropout 0.2
    python3 minigpt.py sample --name base --which best --prompt "Król "
    python3 minigpt.py inspect --name base --prompt "Wawel znajduje się w " --per_head --lens
    python3 minigpt.py eval --name base --ablate_scan

Zależności: `torch`, `numpy` (opcjonalnie `matplotlib` do PNG z attention).
Na RX 6700 XT z ROCm: `export HSA_OVERRIDE_GFX_VERSION=10.3.0`.

Dane (`data/`), checkpointy (`*.pt`) i logi są poza repozytorium (`.gitignore`).
