"""Attack simulation utilities for Carbon Sentinel.

Provides simple byzantine and slow-drift attack generators and a small
simulation runner to show drift accumulation over rounds.
"""
import numpy as np
from typing import List


def byzantine_attack(base_update: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Return a corrupted update by flipping sign and scaling.

    This simulates a maximally disruptive compromised client.
    """
    return -scale * base_update


def slow_drift_attack(base_update: np.ndarray, drift_factor: float = 0.01, round_number: int = 1) -> np.ndarray:
    """Return a subtly poisoned update that drifts gradually over rounds.

    Adds a small bias proportional to `round_number`.
    """
    rng = np.random.default_rng(1234)
    bias = rng.normal(0.0, drift_factor * np.linalg.norm(base_update) * 0.01, size=base_update.shape)
    return base_update + bias * round_number


def simulate_attack_scenario(attack_type: str, n_rounds: int = 10) -> List[float]:
    """Simulate attack evolution and return drift history (L2 norms between rounds).

    Prints per-round stats showing drift accumulation for presentation.
    """
    rng = np.random.default_rng(1)
    dim = 1024
    base = rng.normal(0, 1, size=(dim,))

    drift_history = []
    current = base.copy()

    print(f"Simulating {attack_type} over {n_rounds} rounds")
    for r in range(1, n_rounds + 1):
        if attack_type == "byzantine":
            attack = byzantine_attack(base, scale=5.0) if r == int(n_rounds / 2) else base + rng.normal(0, 0.01, size=(dim,))
        elif attack_type == "slow_drift":
            attack = slow_drift_attack(base, drift_factor=0.02, round_number=r)
        else:
            raise ValueError("Unknown attack_type")

        # simple aggregation: average of honest (base-like) and attack
        # for demo, assume single attacker among many honest clients -> weighted effect small
        # here we approximate new global as average of previous and attack
        new_global = 0.9 * current + 0.1 * attack
        drift = np.linalg.norm(new_global - current)
        drift_history.append(float(drift))
        current = new_global
        print(f"  Round {r:02d}: drift={drift:.6f}")

    return drift_history


if __name__ == "__main__":
    print("--- Byzantine scenario ---")
    b = simulate_attack_scenario("byzantine", n_rounds=10)
    print("--- Slow drift scenario ---")
    s = simulate_attack_scenario("slow_drift", n_rounds=10)
    print("Summary: Byzantine max drift=%.6f, Slow-drift max=%.6f" % (max(b), max(s)))
