"""Data generator for Carbon Sentinel.

Simulates strain sensor readings (62 sensors) for different aircraft profiles
and can inject demo anomalies: crack, overload, drift.

Uses only numpy and scipy.
"""
from typing import List, Tuple
import numpy as np
from scipy import signal


class AircraftDataGenerator:
    """Generate synthetic flight strain data for several aircraft profiles.

    Each flight is an array of shape (n_timesteps, 62), representing a von
    Mises-like strain proxy at 62 sensor locations.
    """

    SENSOR_COUNT = 62

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # Profile templates: (base_freq, base_amp, noise_std, temporal_band)
        self.profiles = {
            "NarrowBodyShortHaul": {
                "freq": 5.0,  # higher-frequency content
                "amp": 0.8,  # low amplitude
                "noise": 0.02,
                "temporal_band": (1, 8),
            },
            "WideBodyLongHaul": {
                "freq": 1.2,  # lower-frequency content
                "amp": 1.6,  # higher amplitude
                "noise": 0.03,
                "temporal_band": (0.5, 3),
            },
            "RepairedAircraft": {
                "freq": 3.0,
                "amp": 1.0,
                "noise": 0.025,
                "temporal_band": (1, 6),
                "asymmetry_gain": 1.25,  # altered stiffness section
            },
        }

    def generate_flight(self, profile: str, n_timesteps: int = 500) -> np.ndarray:
        """Generate a single flight time-series for `profile`.

        Returns:
            flight: np.ndarray shape (n_timesteps, 62)
        """
        if profile not in self.profiles:
            raise ValueError(f"Unknown profile: {profile}")

        p = self.profiles[profile]
        t = np.linspace(0, 1.0, n_timesteps)

        # sensor-wise variation to simulate non-i.i.d behaviour
        sensor_offsets = self.rng.normal(0.0, 0.05, size=(self.SENSOR_COUNT,))
        flight = np.zeros((n_timesteps, self.SENSOR_COUNT), dtype=float)

        for s in range(self.SENSOR_COUNT):
            # per-sensor frequency and amplitude
            freq = p["freq"] * (1.0 + sensor_offsets[s])
            amp = p["amp"] * (1.0 + sensor_offsets[s] * 0.5)

            # combine a few harmonics for realism
            phase = self.rng.uniform(0, 2 * np.pi)
            signal_raw = (
                amp * np.sin(2 * np.pi * freq * t + phase)
                + 0.3 * amp * np.sin(2 * np.pi * freq * 2.3 * t + phase * 0.7)
            )

            # low-pass / smoothing to emulate structural dynamics
            b, a = signal.butter(2, 0.2)
            smooth = signal.filtfilt(b, a, signal_raw)

            flight[:, s] = smooth

        # Add profile-specific asymmetric effect for RepairedAircraft
        if profile == "RepairedAircraft":
            # choose a wing section (sensor block) to modify
            damaged_idx = np.arange(10, 16)
            flight[:, damaged_idx] *= p.get("asymmetry_gain", 1.2)

            # add a small static offset to make asymmetry visible
            flight[:, damaged_idx] += 0.05

        # Add noise
        noise = self.rng.normal(0.0, p["noise"], size=flight.shape)
        flight += noise

        return flight.astype(float)

    def inject_anomaly(self, flight_data: np.ndarray, anomaly_type: str) -> np.ndarray:
        """Inject an anomaly into `flight_data` and return a new array.

        anomaly_type in {"crack", "overload", "drift"}.
        """
        flight = flight_data.copy()
        n_timesteps = flight.shape[0]

        if anomaly_type == "crack":
            # sudden spike in a small cluster of sensors
            t0 = int(self.rng.uniform(0.1, 0.9) * n_timesteps)
            duration = max(1, int(0.01 * n_timesteps))
            cluster_center = self.rng.integers(6, self.SENSOR_COUNT - 6)
            cluster = np.arange(cluster_center - 3, cluster_center + 3)
            spike = 3.0 * np.std(flight) + self.rng.normal(0, 0.1)
            flight[t0 : t0 + duration, cluster] += spike

        elif anomaly_type == "overload":
            # sustained elevated readings across all sensors for a segment
            start = int(self.rng.uniform(0.05, 0.6) * n_timesteps)
            length = int(self.rng.uniform(0.1, 0.4) * n_timesteps)
            boost = 1.5 * np.mean(np.abs(flight))
            flight[start : start + length, :] += boost

        elif anomaly_type == "drift":
            # gradual increase in one wing section over time
            wing_idx = np.arange(2, 8)  # small wing section
            ramp = np.linspace(0, 0.8 * np.mean(np.abs(flight)), n_timesteps)
            # apply to selected sensors
            flight[:, wing_idx] += ramp[:, None]

        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        return flight

    def generate_dataset(self, profile: str, n_flights: int = 50, anomaly_rate: float = 0.1) -> List[Tuple[np.ndarray, int]]:
        """Generate a dataset of flights for `profile`.

        Returns a list of tuples (flight_array, label) where label is 0 (normal) or 1 (anomaly).
        """
        dataset = []
        for i in range(n_flights):
            flight = self.generate_flight(profile)
            if self.rng.random() < anomaly_rate:
                # choose anomaly type randomly
                a_type = self.rng.choice(["crack", "overload", "drift"])
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
        shapes = [f[0].shape for f in ds]
        labels = [f[1] for f in ds]
        all_data = np.stack([f[0] for f in ds], axis=0)  # (n_flights, T, S)
        print(f"Profile: {prof}")
        print(f"  Flights: {len(ds)}, shapes sample: {shapes[0]}")
        print(f"  Labels: {sum(labels)} anomalies out of {len(labels)}")
        print(f"  Global mean: {all_data.mean():.4f}, std: {all_data.std():.4f}\n")
