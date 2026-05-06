"""PyTorch model and training utilities for Carbon Sentinel clients."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class StrainClassifier(nn.Module):
    """Simple 1D CNN classifier for strain time-series.

    Expects input of shape (batch, timesteps, 62). Internally permutes to
    (batch, channels=62, timesteps) for Conv1d layers.
    """

    def __init__(self, n_sensors: int = 62, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_sensors, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, sensors) -> permute to (batch, sensors, timesteps)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)
        out = self.mlp(x)
        return out


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> float:
    """Train model for one epoch and return average loss."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in dataloader:
        X = X.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(1, n)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluate model and return (loss, accuracy)."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).float()
            y = y.to(device).long()
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += X.size(0)
    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    return avg_loss, acc


if __name__ == "__main__":
    print("clients.model: module loaded. Use from clients.model import StrainClassifier")
