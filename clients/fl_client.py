"""Flower NumPyClient implementation for revised single-channel Carbon Sentinel setup."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys
import importlib.util
import logging

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

try:
    from clients.model import StrainClassifier, evaluate, train_one_epoch
    from data.generator import AircraftDataGenerator
    from math.mahalanobis import MahalanobisDetector
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clients.model import StrainClassifier, evaluate, train_one_epoch
    from data.generator import AircraftDataGenerator
    # Use importlib to avoid conflict with builtin math module
    import importlib.util
    spec = importlib.util.spec_from_file_location("mahalanobis_module", str(Path(__file__).resolve().parent.parent / "math" / "mahalanobis.py"))
    if spec and spec.loader:
        mahal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mahal_module)
        MahalanobisDetector = mahal_module.MahalanobisDetector
    else:
        # Fallback: define a dummy detector if import fails
        class MahalanobisDetector:
            def __init__(self):
                pass
            def fit(self, baseline_data):
                pass
            def flag_anomalies(self, test_data, threshold=3.0):
                return np.zeros(test_data.shape[0], dtype=bool)


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
    """Personalized FedAvg client for one aircraft with Mahalanobis pre-filtering."""

    def __init__(self, client_id: str, local_dataset: Dict[str, List[Tuple[np.ndarray, int]]], config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.device = config.get("device", "cpu")

        # Raw datasets (before filtering)
        self.train_ds_raw = LocalDataset(local_dataset.get("train", []))
        self.test_ds = LocalDataset(local_dataset.get("test", []))
        self.model = StrainClassifier().to(self.device)
        
        # Mahalanobis detector for pre-filtering
        self.detector = MahalanobisDetector()
        self.detector_fitted = False
        
        # Filtered dataset (after Mahalanobis filtering)
        self.train_ds = self.train_ds_raw

    def get_parameters(self, config: Dict[str, Any] | None = None) -> List[np.ndarray]:
        return _get_parameters(self.model)

    def _apply_mahalanobis_filtering(self) -> None:
        """Fit Mahalanobis detector on normal data and filter training data."""
        if self.detector_fitted or len(self.train_ds_raw) < 3:  # Need at least 3 samples
            return
        
        try:
            # Extract raw training data
            raw_items = self.train_ds_raw.items
            X_data = np.stack([x for x, _ in raw_items])  # (n, timesteps, 1)
            y_data = np.array([y for _, y in raw_items])
            
            # Flatten to (n, timesteps) for Mahalanobis
            X_flat = X_data.reshape(X_data.shape[0], -1)
            
            # Fit detector on normal data (label=0)
            normal_mask = y_data == 0
            if normal_mask.sum() > 1:
                normal_data = X_flat[normal_mask]
                self.detector.fit(normal_data)
                
                # Flag all data with a conservative threshold
                anomaly_flags = self.detector.flag_anomalies(X_flat, threshold=5.0)  # More conservative
                
                # Keep filtered items, but ensure we don't filter out everything
                filtered_items = [item for item, is_anom in zip(raw_items, anomaly_flags) if not is_anom]
                if len(filtered_items) > 0:  # Only update if we have something left
                    self.train_ds = LocalDataset(filtered_items)
                
                self.detector_fitted = True
        except Exception as e:
            # If filtering fails, just use raw data
            logger.debug(f"Mahalanobis filtering failed: {e}, using raw data")
            self.detector_fitted = True  # Mark as fitted to avoid retrying
            return

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        # 1) Load global parameters.
        _set_parameters(self.model, parameters)
        
        # 2) Apply Mahalanobis pre-filtering (one-time fit)
        self._apply_mahalanobis_filtering()

        # 3) Personalized FedAvg: always fine-tune locally after loading global weights.
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
