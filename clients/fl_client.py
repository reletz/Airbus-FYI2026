"""Flower NumPyClient wrapper for Carbon Sentinel clients.

Implements Personalized FedAvg behavior: client loads global parameters
then fine-tunes locally before returning updated parameters.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import flwr as fl

from clients.model import StrainClassifier, train_one_epoch, evaluate
from data.generator import AircraftDataGenerator


class LocalDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, int]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        X, y = self.items[idx]
        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.long)


def _get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def _set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {}
    for k, arr in zip(keys, parameters):
        tensor = torch.tensor(arr)
        new_state[k] = tensor
    state_dict.update(new_state)
    model.load_state_dict(state_dict)


class CarbonClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, local_dataset: Dict[str, List[Tuple[np.ndarray, int]]], config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.train_ds = LocalDataset(local_dataset.get("train", []))
        self.test_ds = LocalDataset(local_dataset.get("test", []))
        self.device = config.get("device", "cpu")
        self.model = StrainClassifier()

    def get_parameters(self) -> List[np.ndarray]:
        return _get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        # Load global parameters into local model
        _set_parameters(self.model, parameters)

        # Personalization step: fine-tune local model on client's data before returning.
        # This ensures each client personalizes the global model to its local distribution
        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 8))
        lr = float(config.get("lr", 1e-3))

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            train_one_epoch(self.model, train_loader, optimizer, criterion, device=self.device)

        # Return updated parameters and metrics
        updated_params = _get_parameters(self.model)
        num_examples = len(self.train_ds)
        # simple metric placeholder
        metrics = {"num_examples": num_examples}
        return updated_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        # Load global parameters then personalize (fine-tune) before evaluation
        _set_parameters(self.model, parameters)

        # Optionally fine-tune briefly before evaluation to simulate personalization
        local_epochs = int(config.get("local_epochs", 0))
        if local_epochs > 0 and len(self.train_ds) > 0:
            loader = DataLoader(self.train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.get("lr", 1e-3)))
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(local_epochs):
                train_one_epoch(self.model, loader, optimizer, criterion, device=self.device)

        test_loader = DataLoader(self.test_ds, batch_size=int(config.get("batch_size", 8)), shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = evaluate(self.model, test_loader, criterion, device=self.device)
        return float(loss), len(self.test_ds), {"accuracy": float(acc)}


def simulate_client(client_id: str, profile_name: str, n_flights: int = 20, anomaly_rate: float = 0.1, seed: int = 42) -> CarbonClient:
    gen = AircraftDataGenerator(seed=seed)
    ds = gen.generate_dataset(profile_name, n_flights=n_flights, anomaly_rate=anomaly_rate)
    # simple split: 80/20
    split = int(0.8 * len(ds))
    local_dataset = {"train": ds[:split], "test": ds[split:]}
    config = {"device": "cpu"}
    return CarbonClient(client_id, local_dataset, config)


if __name__ == "__main__":
    c = simulate_client("client0", "NarrowBodyShortHaul", n_flights=10)
    print("Simulated client created with train/test sizes:", len(c.train_ds), len(c.test_ds))
