"""Homomorphic encryption utilities for secure model aggregation in Carbon Sentinel.

Uses TenSEAL's CKKS scheme to encrypt and aggregate model weight updates
without decrypting them on the server.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tenseal as ts
except ImportError as exc:
    raise ImportError(
        "tenseal is required for security.encryption. Install it with `pip install tenseal`."
    ) from exc


def setup_tenseal_context() -> Tuple[ts.context, ts.context]:
    """Set up and return (public_context, secret_context) for CKKS encryption.

    The public context is used for encryption (and passed to clients).
    The secret context is used for decryption (kept on a trusted node).

    Returns:
        (public_ctx, secret_ctx) tuple of TenSEAL contexts.
    """
    # Create context with CKKS scheme
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2**40

    # Serialize to get public context (this removes the secret key)
    context_data = context.serialize(save_secret_key=False)
    public_context = ts.context_from(context_data)

    # Keep the full context with secret key for decryption
    secret_context = context

    return public_context, secret_context


def encrypt_update(public_ctx: ts.context, weight_array: np.ndarray) -> ts.CKKSVector:
    """Encrypt a weight update vector using the public context.

    Args:
        public_ctx: TenSEAL public context for encryption.
        weight_array: flat or multi-dimensional numpy array of weights.

    Returns:
        Encrypted weight vector as ts.CKKSVector.
    """
    weight_array = np.asarray(weight_array, dtype=float)
    flat = weight_array.ravel()
    encrypted = ts.ckks_vector(public_ctx, flat.tolist())
    return encrypted


def decrypt_update(secret_ctx: ts.context, encrypted_vector: ts.CKKSVector) -> np.ndarray:
    """Decrypt an encrypted weight update vector using the secret context.

    Args:
        secret_ctx: TenSEAL secret context for decryption.
        encrypted_vector: Encrypted ts.CKKSVector.

    Returns:
        Decrypted weight vector as numpy array.
    """
    # Set the context for decryption
    encrypted_vector.link_context(secret_ctx)
    decrypted = np.array(encrypted_vector.decrypt(), dtype=float)
    return decrypted


def aggregate_encrypted_updates(encrypted_list: list[ts.CKKSVector]) -> ts.CKKSVector:
    """Aggregate encrypted weight updates by summing them homomorphically.

    No decryption needed on the server; addition is performed in the encrypted domain.

    Args:
        encrypted_list: List of encrypted weight vectors (ts.CKKSVector).

    Returns:
        Aggregated (summed) encrypted vector.
    """
    if not encrypted_list:
        raise ValueError("encrypted_list is empty")

    result = encrypted_list[0]
    for enc_vec in encrypted_list[1:]:
        result += enc_vec
    return result


if __name__ == "__main__":
    print("Setting up CKKS context...")
    public_ctx, secret_ctx = setup_tenseal_context()
    print("Context ready")

    # Simulate Client A and Client B each generating a weight update
    print("\nGenerating weight updates...")
    dim = 128
    weight_a = np.random.randn(dim) * 0.01
    weight_b = np.random.randn(dim) * 0.01

    print(f"Weight A shape: {weight_a.shape}, norm: {np.linalg.norm(weight_a):.6f}")
    print(f"Weight B shape: {weight_b.shape}, norm: {np.linalg.norm(weight_b):.6f}")

    # Encrypt both updates using public context
    print("\nEncrypting updates (client-side)...")
    enc_a = encrypt_update(public_ctx, weight_a)
    enc_b = encrypt_update(public_ctx, weight_b)
    print(f"Encrypted A: {type(enc_a)}")
    print(f"Encrypted B: {type(enc_b)}")

    # Server aggregates encrypted updates without decryption
    print("\nAggregating encrypted updates (server-side, no decryption)...")
    enc_sum = aggregate_encrypted_updates([enc_a, enc_b])
    print(f"Encrypted sum: {type(enc_sum)}")

    # Trusted node decrypts the aggregated result
    print("\nDecrypting aggregated result (trusted node)...")
    decrypted_sum = decrypt_update(secret_ctx, enc_sum)

    # Compare with plaintext ground truth
    plaintext_sum = weight_a + weight_b
    error = np.linalg.norm(decrypted_sum - plaintext_sum)
    rel_error = error / (np.linalg.norm(plaintext_sum) + 1e-10)

    print(f"Plaintext A + B norm: {np.linalg.norm(plaintext_sum):.6f}")
    print(f"Decrypted sum norm: {np.linalg.norm(decrypted_sum):.6f}")
    print(f"Absolute error: {error:.6e}")
    print(f"Relative error: {rel_error:.6e}")

    if rel_error < 1e-3:
        print("\n✓ Encryption/decryption successful within CKKS precision")
    else:
        print(f"\n⚠ Warning: relative error {rel_error:.6e} exceeds tolerance")
