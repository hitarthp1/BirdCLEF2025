# BirdCLEF 2025 — Audio Classification (Portfolio Repo)

This repo is a **portfolio-friendly** version of my BirdCLEF project:
- ✅ code + notebooks + explanation
- ❌ **no dataset, no MEL `.npy` feature dumps, no trained weights** (they’re too large / were cleaned up)

If you’re a recruiter: this is meant to show *how I think + how I build pipelines*, not to ship a 20GB repo to your laptop. 🙂

---

## What this project does

BirdCLEF is a bioacoustics challenge: given audio soundscapes, predict which bird species are present.
My pipeline (Kaggle-first workflow):
1. Load 5-second windows from soundscapes
2. Convert audio → mel spectrogram image (single-channel)
3. Run a CNN-based model to output probabilities for **206 classes**
4. Export `submission.csv` in the competition format

---

## Repository layout

```
.
├─ notebooks/
│  ├─ bird-clef-2025-v2.ipynb
│  └─ submission-birdclef.ipynb
├─ src/
│  ├─ model.py          # model skeleton (fill exact head from your final notebook)
│  ├─ preprocess.py     # audio → mel utilities
│  └─ infer.py          # submission.csv generation
├─ configs/
│  └─ class_names.json  # 206 labels extracted from notebook
├─ requirements.txt
└─ .gitignore
```

---

## Why the big files are NOT in GitHub

- **MEL feature `.npy` files** can be huge and don’t belong in git.
- **Weights** can be large and often shouldn’t be committed directly.

Instead, I recommend one of these “grown-up” options:
- **Kaggle Dataset**: upload weights/features there and reference them from the notebook
- **GitHub Releases**: attach `best_model.pth` as a release asset
- **Git LFS**: for weights that must live in git (still not ideal for huge files)
- **DVC** (Data Version Control): best for serious data pipelines

---

## Reproducibility (Kaggle recommended)

This project was built to run on Kaggle paths like:
- `/kaggle/input/birdclef-2025/test_soundscapes`

To reproduce locally you’d need to:
1. Download the dataset (Kaggle)
2. Adjust file paths in the notebooks / scripts

---

## Running inference (when you have weights)

On Kaggle:

```bash
pip install -r requirements.txt
python -m src.infer \
  --test_path /kaggle/input/birdclef-2025/test_soundscapes \
  --weights /kaggle/input/<your-weights-dataset>/best_model.pth \
  --out submission.csv
```

---

## Notes / TODO

- The notebooks contain the full workflow; `src/model.py` is a clean skeleton that you should align with your final trained head (some parts in the notebook were replaced with `...`).
- Add your **CV metric / public LB score** and a screenshot under `assets/` to make this repo look *done*.

---

## Credits

- BirdCLEF dataset & challenge organizers (Kaggle)
- PyTorch / torchaudio / timm
