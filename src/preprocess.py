"""Audio -> mel-spectrogram utilities."""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T


def build_mel_transforms(
    sample_rate: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    f_min: int = 20,
    f_max: int = 16000,
):
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    db = T.AmplitudeToDB()
    return mel, db


def audio_to_mel_image(
    waveform: torch.Tensor,
    sample_rate: int = 32000,
    clip_seconds: int = 5,
):
    """Convert audio (1, N) to mel image tensor (1, 1, n_mels, time).

    This matches the style of the provided submission notebook: split soundscape into 5s windows.
    """
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    mel, db = build_mel_transforms(sample_rate=sample_rate)

    # Ensure length is exactly clip_seconds
    target_len = int(sample_rate * clip_seconds)
    if waveform.size(1) < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
    else:
        waveform = waveform[:, :target_len]

    m = db(mel(waveform))  # (1, n_mels, time)
    m = (m - m.mean()) / (m.std() + 1e-6)  # basic normalization
    return m.unsqueeze(0)  # (1, 1, n_mels, time)
