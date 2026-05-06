"""Single-channel rGO strain data generator for Carbon Sentinel.

MVP assumptions:
- One sensor channel per aircraft flight.
- Strain is simulated directly and returned as shape (timesteps, 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal


class AircraftDataGenerator:
    """Generate single-channel synthetic strain flights for aircraft profiles."""

    GF = 5.64

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.profiles = {
            "NarrowBodyShortHaul": {"amp": 4.0e-4, "freq": 4.0, "noise": 2.0e-5},
            "WideBodyLongHaul": {"amp": 7.0e-4, "freq": 1.6, "noise": 1.5e-5},
            "RepairedAircraft": {"amp": 5.0e-4, "freq": 2.8, "noise": 3.0e-5},
        }

    def generate_flight(self, profile: str, n_timesteps: int = 500) -> np.ndarray:
        """Generate one flight as a strain array shaped (n_timesteps, 1)."""
        if profile not in self.profiles:
            raise ValueError(f"Unknown profile: {profile}")

        p = self.profiles[profile]
        t = np.linspace(0.0, 1.0, n_timesteps)

        phase = self.rng.uniform(0.0, 2.0 * np.pi)
        base = (
            p["amp"] * np.sin(2.0 * np.pi * p["freq"] * t + phase)
            + 0.25 * p["amp"] * np.sin(2.0 * np.pi * (p["freq"] * 2.1) * t + phase * 0.6)
        )

        b, a = signal.butter(2, 0.18)
        smooth = signal.filtfilt(b, a, base)

        if profile == "RepairedAircraft":
            # Slight asymmetry-like behavior in one section of the flight.
            start = int(0.35 * n_timesteps)
            end = int(0.6 * n_timesteps)
            smooth[start:end] *= 1.12

        noisy = smooth + self.rng.normal(0.0, p["noise"], size=n_timesteps)
        return noisy.reshape(-1, 1).astype(float)

    def inject_anomaly(self, flight_data: np.ndarray, anomaly_type: str) -> np.ndarray:
        """Inject crack/overload/drift anomaly into shape (timesteps, 1) data."""
        flight = np.asarray(flight_data, dtype=float).copy()
        if flight.ndim != 2 or flight.shape[1] != 1:
            raise ValueError(f"Expected shape (timesteps, 1), got {flight.shape}")

        n_timesteps = flight.shape[0]

        if anomaly_type == "crack":
            t0 = int(self.rng.uniform(0.1, 0.9) * n_timesteps)
            duration = max(2, int(0.015 * n_timesteps))
            spike = 4.0 * np.std(flight)
            flight[t0 : t0 + duration, 0] += spike
        elif anomaly_type == "overload":
            start = int(self.rng.uniform(0.1, 0.55) * n_timesteps)
            length = int(self.rng.uniform(0.12, 0.35) * n_timesteps)
            level = 1.8 * np.mean(np.abs(flight))
            flight[start : start + length, 0] += level
        elif anomaly_type == "drift":
            ramp = np.linspace(0.0, 1.2 * np.mean(np.abs(flight)), n_timesteps)
            flight[:, 0] += ramp
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

        return flight

    def generate_dataset(
        self,
        profile: str,
        n_flights: int = 50,
        anomaly_rate: float = 0.1,
    ) -> List[Tuple[np.ndarray, int]]:
        """Generate list of `(flight, label)` tuples with binary anomaly labels."""
        dataset: List[Tuple[np.ndarray, int]] = []
        anomaly_types = ["crack", "overload", "drift"]

        for _ in range(n_flights):
            flight = self.generate_flight(profile)
            if self.rng.random() < anomaly_rate:
                a_type = str(self.rng.choice(anomaly_types))
                flight = self.inject_anomaly(flight, a_type)
                label = 1
            else:
                label = 0
            dataset.append((flight, label))

        return dataset


if __name__ == "__main__":
    gen = AircraftDataGenerator(seed=42)
    profiles = ["NarrowBodyShortHaul", "WideBodyLongHaul", "RepairedAircraft"]

    for prof in profiles:
        ds = gen.generate_dataset(prof, n_flights=10, anomaly_rate=0.2)
        labels = np.array([label for _, label in ds], dtype=int)
        stacked = np.stack([flight for flight, _ in ds], axis=0)
        print(f"Profile: {prof}")
        print(f"  flights={len(ds)}, sample_shape={ds[0][0].shape}")
        print(f"  anomalies={int(labels.sum())}/{len(labels)}")
        print(f"  mean={stacked.mean():.6e}, std={stacked.std():.6e}")

    # Save sample CSV from one generated flight
    sample_flight, sample_label = gen.generate_dataset("NarrowBodyShortHaul", n_flights=1, anomaly_rate=0.5)[0]
    out_path = Path("data") / "sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.arange(sample_flight.shape[0], dtype=int)
    labels_col = np.full(sample_flight.shape[0], sample_label, dtype=int)
    csv_data = np.column_stack([timestamps, sample_flight[:, 0], labels_col])
    np.savetxt(
        out_path,
        csv_data,
        delimiter=",",
        header="timestamp,strain,label",
        comments="",
        fmt=["%d", "%.8e", "%d"],
    )
    print(f"Saved sample CSV: {out_path}")
