
"""Small CNN supporting mono and stereo log-Mel inputs."""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 1,
    ):

        super().__init__()

        if in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 or 2")

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)

        x = x.squeeze(-1).squeeze(-1)

        return self.classifier(x)
