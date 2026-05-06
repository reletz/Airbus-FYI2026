## PROMPT 7 — hardware/daq_reader.py

```
Context:
- MVP: 1 rGO sensor channel via Serial (USB)
- Microcontroller sends one float per line (resistance value in Ohm)
- strain formula: epsilon = (R - R0) / (GF * R0), GF=5.64
- Output must match generator.py shape: (buffer_size, 1)

Create: hardware/daq_reader.py

Class SensorDAQ:
- __init__(port, baud_rate=9600, gauge_factor=5.64, base_resistance=1000.0)
- calibrate(n_samples=100): read n_samples at no-load, set self.R0 as mean
- read_live_stream(buffer_size=500) → np.ndarray (buffer_size, 1)
  - handle SerialException and non-float lines with logging, not crash
- log_to_csv(buffer, filepath): save timestamp + R_ohm + strain to CSV
  - columns: timestamp, R_ohm, strain
- Use: numpy, pyserial

__main__ block:
- DummySerial class simulating single-channel resistance values
- calibrate() → read_live_stream(100) → log_to_csv()
- Print shape + first 3 rows

Next: read_live_stream() output is drop-in replacement for
generator.py generate_flight() output.
```

---

## PROMPT 8 — math/ifem.py

```
Context:
- Sensor output: np.ndarray (62,) per timestep
- Goal: reconstruct 2D displacement field from discrete strain measurements
- Use simplified iFEM / iKS4: least-squares via pseudo-inverse transfer matrix

Create: math/ifem.py

Class ShapeSensing_iFEM:
- __init__(): build transfer matrix T (pseudo-inverse) mapping (62,) → (10,10) grid
  - mock sensor coords as uniform 2D grid
- reconstruct_displacement(strain_array: np.ndarray shape (62,)) → np.ndarray (10,10)
  - U = T @ strain_array, reshape to (10,10)
- Use: numpy only

__main__ block:
- Generate random (62,) strain input
- Plot reconstructed (10,10) field with matplotlib imshow
- Save figure to math/ifem_test.png

Next: dashboard.py Tab "iFEM Shape Sensing" calls reconstruct_displacement()
and renders output as Plotly 3D Surface (see PROMPT 11).
```

---

## PROMPT 9 — math/mahalanobis.py

```
Context:
- Complement to CNN in fl_client.py
- Detects BVID (Barely Visible Impact Damage) statistically before DL inference
- Input shape matches generator.py output: (n_samples, 62)

Create: math/mahalanobis.py

Class MahalanobisDetector:
- fit(baseline_data: np.ndarray (n, 62)): store mean_, inv_cov_ (use np.linalg.pinv)
- predict_score(test_data: np.ndarray (m, 62)) → np.ndarray (m,) of distances
- flag_anomalies(test_data, threshold) → np.ndarray bool (m,)
- Use: numpy only

__main__ block:
- fit on 200 normal samples from generator.py
- flag_anomalies on 50 samples (10 injected anomalies)
- Print precision and recall

Add comment in math/mahalanobis.py (not in fl_client.py):
# Integration point: in CarbonClient.fit(), call detector.flag_anomalies()
# before passing data to StrainClassifier to pre-filter outliers.

Next: detector instance saved via pickle; loaded in fl_client.py at runtime.
```

---

## PROMPT 10 — security/encryption.py

```
Context:
- FL clients send model weight updates to server (see fl_client.py get_parameters())
- Encrypt weight vectors before transmission using CKKS homomorphic encryption
- Server aggregates encrypted vectors without decrypting
- Use: tenseal

Create: security/encryption.py

Functions:
1. setup_tenseal_context() → (public_ctx, secret_ctx)
   - CKKS scheme, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60]

2. encrypt_update(public_ctx, weight_array: np.ndarray) → ts.CKKSVector
   - flatten weight_array before encrypting

3. decrypt_update(secret_ctx, encrypted_vector: ts.CKKSVector) → np.ndarray

4. aggregate_encrypted_updates(encrypted_list: list[ts.CKKSVector]) → ts.CKKSVector
   - sum all vectors in-place (HE addition, no decryption)

__main__ block:
- Client A and B each encrypt a random (128,) weight vector
- Server calls aggregate_encrypted_updates([enc_A, enc_B])
- Client decrypts aggregate, prints: original A+B vs decrypted (should match within CKKS error)

Next: in run_demo.py, wrap fl_client get_parameters() output with encrypt_update()
before passing to fltrust_aggregate() (see PROMPT 11).
```

---

## PROMPT 11 — Update run_demo.py + dashboard.py

```
Context:
- All modules exist: generator.py, model.py, fl_client.py, fl_server.py,
  fltrust.py, attack_sim.py, ifem.py, mahalanobis.py, encryption.py
- run_demo.py already simulates FL loop and saves demo/results.json
- dashboard.py already has sections: Fleet Overview, Training Progress,
  Anomaly Detection, Security Monitor

Task A — run_demo.py:
Add encryption step inside the FL round loop (after local fine-tuning):
  enc_update = encrypt_update(public_ctx, flat_weights)
  # For FLTrust compatibility: decrypt at aggregator node before trust scoring
  dec_update = decrypt_update(secret_ctx, enc_update)
  trust_scores = compute_trust_scores([dec_update, ...], trusted_update)
Add --no-encrypt flag (argparse) to skip encryption for speed during dev.
Output both encrypted and plaintext norm to results.json for dashboard.

Task B — dashboard.py:
Add new tab "iFEM Shape Sensing" (use st.tabs()):
- Input: slider selecting timestep from loaded flight data
- Call ifem.reconstruct_displacement(strain_at_timestep)
- Render output (10,10) as plotly go.Surface with colorscale="RdBu"
- Add caption: "Reconstructed wing displacement field at timestep N"

Provide only the modified blocks (the FL loop in run_demo.py and
the new tab block in dashboard.py), not full rewrites of both files.
```