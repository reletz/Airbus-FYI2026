"""Flower NumPyClient implementation for revised single-channel Carbon Sentinel setup."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from clients.model import StrainClassifier, evaluate, train_one_epoch
    from data.generator import AircraftDataGenerator
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clients.model import StrainClassifier, evaluate, train_one_epoch
    from data.generator import AircraftDataGenerator


class LocalDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        X, y = self.items[idx]
        return torch.from_numpy(X).float(), torch.tensor(float(y), dtype=torch.float32)


def _get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [param.detach().cpu().numpy() for _, param in model.state_dict().items()]


def _set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {}
    for key, arr in zip(keys, parameters):
        new_state[key] = torch.tensor(arr)
    state_dict.update(new_state)
    model.load_state_dict(state_dict)


class CarbonClient(fl.client.NumPyClient):
    """Personalized FedAvg client for one aircraft."""

    def __init__(self, client_id: str, local_dataset: Dict[str, List[Tuple[np.ndarray, int]]], config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.device = config.get("device", "cpu")

        self.train_ds = LocalDataset(local_dataset.get("train", []))
        self.test_ds = LocalDataset(local_dataset.get("test", []))
        self.model = StrainClassifier().to(self.device)

    def get_parameters(self, config: Dict[str, Any] | None = None) -> List[np.ndarray]:
        return _get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        # 1) Load global parameters.
        _set_parameters(self.model, parameters)

        # 2) Personalized FedAvg: always fine-tune locally after loading global weights.
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 8))
        lr = float(config.get("lr", 1e-3))

        loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loss = 0.0
        for _ in range(local_epochs):
            train_loss = train_one_epoch(self.model, loader, optimizer, criterion, device=self.device)

        updated_params = _get_parameters(self.model)
        num_examples = len(self.train_ds)
        metrics = {"train_loss": float(train_loss)}
        return updated_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        _set_parameters(self.model, parameters)

        batch_size = int(config.get("batch_size", 8))
        test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()

        loss, acc, _, _ = evaluate(self.model, test_loader, criterion=criterion, device=self.device)
        return float(loss), len(self.test_ds), {"accuracy": float(acc)}


def simulate_client(
    client_id: str,
    profile_name: str,
    n_flights: int = 20,
    anomaly_rate: float = 0.1,
    seed: int = 42,
) -> CarbonClient:
    gen = AircraftDataGenerator(seed=seed)
    ds = gen.generate_dataset(profile_name, n_flights=n_flights, anomaly_rate=anomaly_rate)

    split = max(1, int(0.8 * len(ds)))
    local_dataset = {"train": ds[:split], "test": ds[split:]}
    return CarbonClient(client_id, local_dataset, config={"device": "cpu"})


if __name__ == "__main__":
    client = simulate_client("client0", "NarrowBodyShortHaul", n_flights=10)
    print("Simulated client sizes:", len(client.train_ds), len(client.test_ds))
