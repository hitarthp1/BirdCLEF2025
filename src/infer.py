"""Inference script skeleton to create a BirdCLEF-style submission.csv.

This is designed to be run on Kaggle (recommended) where the dataset paths exist.

Usage (Kaggle):
  python -m src.infer --test_path /kaggle/input/birdclef-2025/test_soundscapes --weights /kaggle/input/<your-dataset>/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from .model import EffB3ResNetEnsemble
from .preprocess import audio_to_mel_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_path", type=str, required=True, help="Folder with .ogg soundscapes.")
    p.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth).")
    p.add_argument("--meta_json", type=str, default=None, help="Optional metadata json with keys latitude/longitude/rating.")
    p.add_argument("--class_names", type=str, default="configs/class_names.json")
    p.add_argument("--out", type=str, default="submission.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    with open(args.class_names, "r") as f:
        class_names = json.load(f)

    # Default metadata (matches your notebook pattern: mean lat/long/rating)
    meta = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    if args.meta_json:
        with open(args.meta_json, "r") as f:
            m = json.load(f)
        meta = torch.tensor([[m.get("latitude", 0.0), m.get("longitude", 0.0), m.get("rating", 0.0)]], dtype=torch.float32)

    model = EffB3ResNetEnsemble(num_classes=len(class_names), metadata_dim=meta.size(1))
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    files = sorted([f for f in os.listdir(args.test_path) if f.endswith(".ogg")])
    submission_rows = []

    for fname in tqdm(files):
        waveform, sr = torchaudio.load(os.path.join(args.test_path, fname))
        if sr != 32000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=32000)

        total_seconds = waveform.size(1) // 32000
        for start_sec in range(0, max(0, total_seconds - 4), 5):
            clip = waveform[:, start_sec*32000:(start_sec+5)*32000]
            x_img = audio_to_mel_image(clip, sample_rate=32000, clip_seconds=5)  # (1,1,mel,time)
            with torch.no_grad():
                logits = model(x_img, meta)
                probs = torch.sigmoid(logits)[0].numpy()

            row_id = f"{fname.replace('.ogg','')}_{start_sec + 5}"
            row = {"row_id": row_id}
            row.update({class_names[i]: float(probs[i]) for i in range(len(class_names))})
            submission_rows.append(row)

    sub = pd.DataFrame(submission_rows)
    sub = sub[["row_id"] + class_names]
    sub.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(sub)} rows.")


if __name__ == "__main__":
    main()
