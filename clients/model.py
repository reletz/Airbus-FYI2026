"""PyTorch model and training utilities for single-channel Carbon Sentinel clients."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class StrainClassifier(nn.Module):
    """Binary classifier for strain sequences shaped (batch, timesteps, 1)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, timesteps, 1) -> (batch, 1, timesteps)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits.squeeze(1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> float:
    """Train one epoch and return average BCE loss."""
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    total_loss = 0.0
    n = 0

    for X, y in dataloader:
        X = X.to(device).float()
        y = y.to(device).float()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / max(1, n)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[float, float, float, float]:
    """Evaluate and return (loss, accuracy, precision, recall)."""
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss = 0.0
    n = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).float()
            y = y.to(device).float()
            logits = model(X)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size

            tp += int(((preds == 1) & (y == 1)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())

    avg_loss = total_loss / max(1, n)
    accuracy = (tp + tn) / max(1, n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return avg_loss, accuracy, precision, recall


if __name__ == "__main__":
    print("clients.model ready (single-channel BCE classifier)")
