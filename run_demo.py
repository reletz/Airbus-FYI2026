"""Integration script to run a local federated learning demo for Carbon Sentinel.

Runs a simplified Personalized FedAvg loop locally without a Flower server.
Saves results to `demo/results.json` for the dashboard.
"""
import argparse
import json
import logging
from typing import List, Tuple

import numpy as np
import torch
import yaml

try:
    from clients.model import StrainClassifier
    from clients.fl_client import simulate_client, _get_parameters as client_get_parameters
except Exception as e:
    raise ImportError("Missing clients module. Run from repository root and ensure clients/ exists.") from e

try:
    from security.fltrust import compute_trust_scores, fltrust_aggregate
    from security.attack_sim import byzantine_attack
except Exception as e:
    raise ImportError("Missing security module. Ensure security/ exists.") from e

from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_demo")


def params_list_to_flat(params: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, Tuple[int, ...]]]]:
    """Flatten a list of numpy arrays and return flat vector plus shapes metadata."""
    flat_parts = []
    shapes = []
    for a in params:
        arr = np.asarray(a)
        shapes.append((arr.size, arr.shape))
        flat_parts.append(arr.ravel())
    if flat_parts:
        flat = np.concatenate(flat_parts)
    else:
        flat = np.array([])
    return flat, shapes


def flat_to_params_list(flat: np.ndarray, shapes: List[Tuple[int, Tuple[int, ...]]]) -> List[np.ndarray]:
    parts = []
    idx = 0
    for size, shape in shapes:
        part = flat[idx : idx + size].reshape(shape)
        parts.append(part)
        idx += size
    return parts


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_demo(attack: bool = False, cfg_path: str = "config.yaml"):
    cfg = load_config(cfg_path)
    n_clients = int(cfg.get("number_of_clients", 5))
    n_rounds = int(cfg.get("fl_rounds", 10))
    anomaly_threshold = float(cfg.get("anomaly_threshold", 2.5))

    # Create simulated clients
    clients = [simulate_client(f"AC_{i}", "NarrowBodyShortHaul", n_flights=20, anomaly_rate=0.15, seed=42 + i) for i in range(n_clients)]

    # Initialize global model
    device = "cpu"
    global_model = StrainClassifier()
    global_model.to(device)

    # Extract initial global parameters as numpy arrays
    global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
    global_flat, shapes = params_list_to_flat(global_params)

    drift_history = []
    rounds_flagged = []
    trust_history = []
    low_trust_counts = {c.client_id: 0 for c in clients}

    for r in range(1, n_rounds + 1):
        logger.info("Round %d/%d", r, n_rounds)
        # Each client fine-tunes locally starting from global params
        client_updates = []
        client_weights = []
        client_infos = []

        for i, c in enumerate(clients):
            # clients in this code expect full param list as input
            updated_params, num_examples, metrics = c.fit(list(global_params), {"local_epochs": 1, "batch_size": 8, "lr": 1e-3})

            # compute update = updated - global
            updated_flat, _ = params_list_to_flat(updated_params)
            update = updated_flat - global_flat

            # if attack and choose client 0 as byzantine, corrupt its update
            if attack and i == 0:
                update = byzantine_attack(update, scale=10.0)

            client_updates.append(update)
            client_weights.append(num_examples)
            client_infos.append({"client_id": c.client_id, "num_examples": num_examples, **(metrics or {})})

        client_updates = np.stack(client_updates, axis=0)

        # Trusted update: for demo use mean of client updates excluding extreme norms
        norms = np.linalg.norm(client_updates, axis=1)
        median = np.median(norms)
        # pick trusted indices with norm <= 2*median
        trusted_idx = norms <= (2.0 * median + 1e-8)
        if trusted_idx.sum() == 0:
            trusted_update = client_updates.mean(axis=0)
        else:
            trusted_update = client_updates[trusted_idx].mean(axis=0)

        scores = compute_trust_scores([u for u in client_updates], trusted_update)
        trust_history.append(scores)

        # Flag clients with low trust
        flagged = [i for i, s in enumerate(scores) if s < 0.3]
        if flagged:
            rounds_flagged.append(r)
            logger.warning("Round %d: flagged clients %s", r, flagged)

        for i, score in enumerate(scores):
            if score < 0.3:
                low_trust_counts[clients[i].client_id] += 1

        # Aggregate using FLTrust weighted average of updates
        agg_update = fltrust_aggregate([u for u in client_updates], scores)

        # Update global parameters
        new_global_flat = global_flat + agg_update

        # Compute drift and log if above threshold
        drift = float(np.linalg.norm(new_global_flat - global_flat))
        drift_history.append(drift)
        if drift > anomaly_threshold:
            logger.warning("Baseline drift detected in round %d: drift=%.6f", r, drift)

        # Set new global params
        new_params = flat_to_params_list(new_global_flat, shapes)
        global_params = new_params
        global_flat = new_global_flat

    # After training: evaluate per-client final accuracy
    final_accuracies = {}
    for c in clients:
        loss, n_examples, metrics = c.evaluate(list(global_params), {"local_epochs": 0, "batch_size": 8})
        final_accuracies[c.client_id] = metrics.get("accuracy", None)

    results = {
        "drift_history": drift_history,
        "rounds_flagged": rounds_flagged,
        "trust_history": trust_history,
        "final_accuracies": final_accuracies,
        "low_trust_counts": low_trust_counts,
    }

    out_path = Path("demo") / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    consistently_low_trust = [client_id for client_id, count in low_trust_counts.items() if count >= max(1, n_rounds // 3)]

    print("Final report")
    print("Per-aircraft final accuracy:")
    for client_id, accuracy in final_accuracies.items():
        print(f"  {client_id}: {accuracy:.4f}")
    print(f"Rounds flagged for drift: {rounds_flagged}")
    print(f"Consistently low-trust clients: {consistently_low_trust if consistently_low_trust else 'none'}")
    print("Run complete. Results saved to", str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", action="store_true", help="Inject one Byzantine client for security demo")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_demo(attack=args.attack, cfg_path=args.config)


if __name__ == "__main__":
    main()
