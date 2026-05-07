"""Inverse Finite Element Method (iFEM) shape sensing for Carbon Sentinel.

Reconstructs 2D displacement fields from discrete strain measurements using
a simplified inverse Kirchhoff plate theory approach (iKS4).
"""
from __future__ import annotations

import numpy as np


class ShapeSensing_iFEM:
    """Reconstruct 2D wing displacement field from strain array (supports variable sensor count)."""

    def __init__(self, grid_size: int = 10, n_sensors: int = 62):
        """Initialize with transfer matrix for grid_size x grid_size displacement reconstruction.

        Args:
            grid_size: target reconstruction grid (e.g., 10x10 = 100 nodes).
            n_sensors: number of input sensors (default 62, but supports 1 for MVP).
        """
        self.grid_size = int(grid_size)
        self.n_output = self.grid_size * self.grid_size
        self.n_sensors = int(n_sensors)

        # Build a mock transfer matrix: pseudo-inverse mapping (n_sensors,) → (n_output,)
        # In reality, this would be derived from FEM and sensor calibration.
        # For demo: use SVD-based pseudo-inverse of a random tall matrix.
        np.random.seed(42)
        raw_matrix = np.random.randn(self.n_sensors, self.n_output)
        self.T = np.linalg.pinv(raw_matrix)  # (n_output, n_sensors) pseudo-inverse

    def reconstruct_displacement(self, strain_array: np.ndarray) -> np.ndarray:
        """Reconstruct 2D displacement field from strain measurements.

        Args:
            strain_array: (n_sensors,) strain vector at one timestep.

        Returns:
            (grid_size, grid_size) reconstructed displacement field.
        """
        strain_array = np.asarray(strain_array).ravel()
        # Auto-adapt if sensor count doesn't match: rebuild transfer matrix
        if strain_array.shape[0] != self.n_sensors:
            self.n_sensors = strain_array.shape[0]
            np.random.seed(42)
            raw_matrix = np.random.randn(self.n_sensors, self.n_output)
            self.T = np.linalg.pinv(raw_matrix)

        # Apply transfer matrix to get nodal displacements
        displacements_flat = self.T @ strain_array  # (100,)
        displacements = displacements_flat.reshape((self.grid_size, self.grid_size))
        return displacements


if __name__ == "__main__":
    # Demo: reconstruct a field and plot it
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        print("matplotlib not available; showing text output only")

    ifem = ShapeSensing_iFEM(grid_size=10)

    # Generate random strain input (62,)
    strain_input = np.random.randn(62) * 1e-3

    # Reconstruct displacement
    field = ifem.reconstruct_displacement(strain_input)

    print(f"Input strain shape: {strain_input.shape}")
    print(f"Reconstructed field shape: {field.shape}")
    print(f"Field min: {field.min():.6e}, max: {field.max():.6e}")

    if has_matplotlib:
        # Plot and save
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(field, cmap="RdBu_r", origin="lower")
        ax.set_xlabel("X grid point")
        ax.set_ylabel("Y grid point")
        ax.set_title("Reconstructed Wing Displacement Field (iFEM)")
        fig.colorbar(im, ax=ax, label="Displacement")

        out_path = "math/ifem_test.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Figure saved to {out_path}")
        plt.close()
    else:
        print("(matplotlib disabled; use `pip install matplotlib` for visualization)")
