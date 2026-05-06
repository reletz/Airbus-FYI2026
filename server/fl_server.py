"""Flower server strategy with drift monitoring for Carbon Sentinel.

This module defines `CarbonStrategy`, a small wrapper around Flower's FedAvg
strategy that computes the L2 norm change in global model weights after each
round (drift). Slow, systematic poisoning/drift attacks produce small per-round
changes that accumulate; tracking per-round drift helps detect such attacks
that plain FedAvg aggregation would not flag.
"""
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters


logger = logging.getLogger(__name__)


class CarbonStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that records per-round drift in global weights.

    The strategy computes the L2 norm between the flattened new global
    parameters and the previous round's parameters and stores these values in
    `drift_history`. If a drift exceeds `drift_threshold`, a WARNING is logged.
    """

    def __init__(self, drift_threshold: float = 1e-3, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.drift_history: List[float] = []
        self.prev_flat: Optional[np.ndarray] = None
        self.drift_threshold = float(drift_threshold)

    def aggregate_fit(self, rnd: int, results, failures):
        # Call parent FedAvg aggregation
        aggregated = super().aggregate_fit(rnd, results, failures)

        # The parent may return None when aggregation can't be performed
        if aggregated is None:
            return None

        # aggregated can be either a Parameters object or a tuple (Parameters, dict)
        if isinstance(aggregated, tuple) and len(aggregated) >= 1:
            params = aggregated[0]
            rest = aggregated[1] if len(aggregated) > 1 else None
        elif isinstance(aggregated, Parameters):
            params = aggregated
            rest = None
        else:
            # Unexpected type: pass through
            return aggregated

        # Convert to ndarrays and flatten into a single vector
        try:
            nds = parameters_to_ndarrays(params)
        except Exception:
            # If conversion fails, skip drift computation but return aggregated result
            logger.exception("Failed to convert aggregated parameters to ndarrays for drift computation")
            return aggregated

        flat = np.concatenate([x.ravel() for x in nds]) if len(nds) > 0 else np.array([])

        if self.prev_flat is not None and flat.size == self.prev_flat.size:
            drift = float(np.linalg.norm(flat - self.prev_flat))
            self.drift_history.append(drift)
            if drift > self.drift_threshold:
                logger.warning("Baseline drift detected in round %d — possible slow drift attack (drift=%.6f)", rnd, drift)
        else:
            # First round or shape mismatch -> record 0.0
            self.drift_history.append(0.0)

        # Save for next round
        self.prev_flat = flat.copy()

        # Recompose return value to match parent return format
        if rest is not None:
            return params, rest
        return params


def get_drift_report(drift_history: List[float], threshold: float = 0.0) -> Dict[str, Any]:
    """Compute a simple drift report.

    Returns max, mean, and list of rounds (1-based) where drift exceeded threshold.
    """
    if not drift_history:
        return {"max_drift": 0.0, "mean_drift": 0.0, "rounds_flagged": []}
    arr = np.array(drift_history, dtype=float)
    flagged = [i + 1 for i, v in enumerate(arr) if v > threshold]
    return {"max_drift": float(arr.max()), "mean_drift": float(arr.mean()), "rounds_flagged": flagged}


def run_server(config: Dict[str, Any]):
    """Start a Flower server using CarbonStrategy and print drift summary afterward.

    Args:
        config: dict containing at least `fl_rounds` (int) and optional `drift_threshold` (float).
    """
    rounds = int(config.get("fl_rounds", 10))
    drift_threshold = float(config.get("drift_threshold", config.get("anomaly_threshold", 2.5)))

    # Initialize strategy with drift threshold
    strategy = CarbonStrategy(drift_threshold=drift_threshold)

    # Start Flower server (will block until finished)
    server_config = fl.server.ServerConfig(num_rounds=rounds)
    logger.info("Starting Flower server for %d rounds", rounds)
    fl.server.start_server(server_address="0.0.0.0:8080", config=server_config, strategy=strategy)

    # After training completes, print drift summary
    report = get_drift_report(strategy.drift_history, drift_threshold)
    print("Drift history summary:")
    print(f"  max_drift: {report['max_drift']:.6f}")
    print(f"  mean_drift: {report['mean_drift']:.6f}")
    print(f"  rounds_flagged: {report['rounds_flagged']}")


"""
Why drift monitoring helps
-------------------------
Slow poisoning attacks add a tiny bias to client updates each round. FedAvg's
aggregation may not consider these tiny changes anomalous in any single round,
but tracking the L2 change in global weights over rounds reveals a consistent
trend (small positive drift accumulating). Flagging rounds with unusually high
drift helps detect these slow attacks that evade per-round outlier detection.
"""
