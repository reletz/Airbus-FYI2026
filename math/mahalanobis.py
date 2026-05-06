"""Mahalanobis distance-based anomaly detection for Carbon Sentinel.

Detects Barely Visible Impact Damage (BVID) and other statistical outliers
in strain measurements using covariance-based distance metrics.
"""
from __future__ import annotations

import numpy as np


class MahalanobisDetector:
    """Statistical anomaly detector using Mahalanobis distance.

    Detects outliers in multivariate strain data by computing the distance
    of each sample from the baseline (normal) distribution.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None

    def fit(self, baseline_data: np.ndarray) -> None:
        """Fit detector on baseline (normal) strain data.

        Args:
            baseline_data: (n_samples, 62) array of normal strain measurements.
        """
        baseline_data = np.asarray(baseline_data, dtype=float)
        if baseline_data.ndim != 2 or baseline_data.shape[1] != 62:
            raise ValueError(f"Expected shape (n, 62), got {baseline_data.shape}")

        self.mean_ = baseline_data.mean(axis=0)
        cov = np.cov(baseline_data.T)  # (62, 62) covariance matrix
        self.inv_cov_ = np.linalg.pinv(cov)  # pseudo-inverse for numerical stability

    def predict_score(self, test_data: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each test sample.

        Args:
            test_data: (n_samples, 62) strain measurements.

        Returns:
            (n_samples,) array of Mahalanobis distances (lower = more normal).
        """
        if self.mean_ is None or self.inv_cov_ is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        test_data = np.asarray(test_data, dtype=float)
        if test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)
        if test_data.shape[1] != 62:
            raise ValueError(f"Expected shape (n, 62), got {test_data.shape}")

        diff = test_data - self.mean_  # (n, 62)
        distances = np.sqrt(np.sum(diff @ self.inv_cov_ * diff, axis=1))
        return distances

    def flag_anomalies(self, test_data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Flag samples as anomalous based on Mahalanobis distance threshold.

        Args:
            test_data: (n_samples, 62) strain measurements.
            threshold: Mahalanobis distance threshold for flagging (default 3.0 sigma).

        Returns:
            (n_samples,) boolean array; True means anomalous.
        """
        scores = self.predict_score(test_data)
        return scores > threshold


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generator import AircraftDataGenerator

    # Fit detector on normal data
    gen = AircraftDataGenerator(seed=42)
    normal_flights = gen.generate_dataset("NarrowBodyShortHaul", n_flights=200, anomaly_rate=0.0)
    normal_data = np.stack([f[0] for f in normal_flights], axis=0)  # (200, 500, 62)
    # Flatten to (200*500, 62)
    normal_data_flat = normal_data.reshape(-1, 62)

    detector = MahalanobisDetector()
    detector.fit(normal_data_flat)
    print(f"Detector fitted on {normal_data_flat.shape[0]} normal samples")

    # Test on mixed data: normal + injected anomalies
    test_flights = []
    for i in range(50):
        flight = gen.generate_flight("NarrowBodyShortHaul")
        if i < 10:
            # Inject anomaly in first 10 samples
            anomaly_type = ["crack", "overload", "drift"][i % 3]
            flight = gen.inject_anomaly(flight, anomaly_type)
        test_flights.append(flight)

    test_data = np.vstack(test_flights)  # (50*500, 62)
    flags = detector.flag_anomalies(test_data, threshold=3.0)

    # Compute precision and recall
    # Ground truth: first 10*500 are anomalies, rest are normal
    n_test_anomaly = 10 * 500
    tp = np.sum(flags[:n_test_anomaly])
    fp = np.sum(flags[n_test_anomaly:])
    fn = n_test_anomaly - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

    # Integration note:
    # In CarbonClient.fit(), call detector.flag_anomalies() on local training data
    # before passing to StrainClassifier to pre-filter statistical outliers.
