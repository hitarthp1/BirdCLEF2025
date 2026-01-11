"""Model definitions for BirdCLEF.

Note:
- This repo intentionally does NOT include trained weights.
- In the provided notebook, parts of the model head were replaced with '...'.
  Fill those sections from your original Kaggle notebook (recommended),
  then export the complete code here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from timm import create_model


class EffB3ResNetEnsemble(nn.Module):
    """EfficientNet-B3 + ResNet-18 image-backbone ensemble + metadata MLP.

    Inputs:
      x: (B, 1, H, W) mel-spectrogram-like image
      meta: (B, metadata_dim)
    Output:
      logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 206, metadata_dim: int = 3):
        super().__init__()

        self.effb3 = create_model(
            "efficientnet_b3",
            pretrained=False,
            in_chans=1,
            num_classes=0,
        )

        self.resnet18 = tv_models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()

        # TODO: replace with your exact metadata head from training
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # TODO: replace with your exact classifier head from training
        # EfficientNet-B3 feature dim is typically 1536, ResNet-18 is 512, plus 128 meta = 2176
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536 + 512 + 128, num_classes),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        feat_effb3 = self.effb3(x)
        feat_resnet = self.resnet18(x)
        feat_meta = self.metadata_head(meta)
        combined = torch.cat([feat_effb3, feat_resnet, feat_meta], dim=1)
        return self.classifier(combined)
