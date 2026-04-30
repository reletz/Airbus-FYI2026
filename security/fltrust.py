"""FLTrust utilities for Carbon Sentinel.

Provides trust score computation and FLTrust-style aggregation for flattened
model updates.
"""
from typing import List
import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1:
        a = a.ravel()
    if b.ndim != 1:
        b = b.ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_trust_scores(client_updates: List[np.ndarray], trusted_update: np.ndarray) -> List[float]:
    """Compute FLTrust trust scores (ReLU of cosine similarity).

    Args:
        client_updates: list of flattened numpy arrays (one per client)
        trusted_update: flattened numpy array from server's trusted data

    Returns:
        List of trust scores between 0 and 1 (negative similarities clipped to 0).
    """
    scores = []
    for upd in client_updates:
        sim = _cosine_similarity(upd, trusted_update)
        score = max(0.0, sim)  # ReLU: negative similarity -> 0 trust
        # Numerical safety
        score = float(np.clip(score, 0.0, 1.0))
        scores.append(score)
    return scores


def fltrust_aggregate(client_updates: List[np.ndarray], trust_scores: List[float]) -> np.ndarray:
    """Aggregate client updates weighted by trust_scores.

    If all trust scores are zero, falls back to simple mean of updates.
    """
    weights = np.array(trust_scores, dtype=float)
    updates = np.stack([u.ravel() for u in client_updates], axis=0)  # (n_clients, dim)
    if weights.sum() <= 0.0:
        agg = updates.mean(axis=0)
    else:
        w = weights / (weights.sum())
        agg = (w[:, None] * updates).sum(axis=0)
    return agg


def run_fltrust_demo(n_clients: int = 5, n_byzantine: int = 1):
    """Demo: simulate honest and byzantine client updates and run FLTrust.

    Prints each client's trust score, flagged clients (score < 0.3), and
    aggregated result norm for presentation.
    """
    rng = np.random.default_rng(42)

    dim = 1024
    base = rng.normal(0, 1, size=(dim,))  # canonical base update

    client_updates = []
    for i in range(n_clients):
        if i < n_byzantine:
            # malicious: large random vector far from base
            upd = rng.normal(0, 10.0, size=(dim,))
        else:
            # honest: base + small noise
            upd = base + rng.normal(0, 0.05, size=(dim,))
        client_updates.append(upd)

    # server trusted update (from small trusted dataset) -> similar to base
    trusted_update = base + rng.normal(0, 0.01, size=(dim,))

    scores = compute_trust_scores(client_updates, trusted_update)
    flagged = [i for i, s in enumerate(scores) if s < 0.3]

    print("FLTrust demo")
    for i, s in enumerate(scores):
        tag = "(BYZ)" if i < n_byzantine else ""
        print(f"  Client {i:02d} {tag}: trust_score={s:.3f}")
    if flagged:
        print(f"  Flagged clients (score<0.3): {flagged}")
    else:
        print("  No clients flagged")

    agg = fltrust_aggregate(client_updates, scores)
    print(f"  Aggregated update L2 norm: {np.linalg.norm(agg):.4f}")


if __name__ == "__main__":
    run_fltrust_demo(n_clients=6, n_byzantine=1)
