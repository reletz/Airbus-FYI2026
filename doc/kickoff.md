# Carbon Sentinel 
---

## PROMPT 0 — Repo Kickoff

```
You are a senior Python engineer helping set up a research demo project called Carbon Sentinel.

Carbon Sentinel is a Federated Learning system for aircraft structural health monitoring. It detects anomalies in aircraft strain sensor data using a privacy-preserving, attack-resilient federated learning pipeline.

Create a complete project scaffold with the following structure:

./
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   └── (empty, with .gitkeep)
├── clients/
│   └── (empty, with .gitkeep)
├── server/
│   └── (empty, with .gitkeep)
├── security/
│   └── (empty, with .gitkeep)
└── demo/
    └── (empty, with .gitkeep)

For requirements.txt, include: flwr, torch, tenseal, streamlit, plotly, numpy, scipy, scikit-learn, pyyaml.

For config.yaml, include: number of clients (default 5), number of FL rounds (default 10), mu for FedProx (default 0.01), anomaly threshold (default 2.5), and random seed (default 42).

For README.md, write a brief description of the project covering: what it does, the three pillars (rGO sensors, iFEM math, FL system), and how to install and run it.

Output each file's full content clearly labeled.
```

---

## PROMPT 1 — Module 1: Data Generator

```
Context:
- MVP: 1 rGO sensor channel (not 62)
- Physical measurement: R0_ohm, R_ohm, load_N, temperature_C, humidity_pct
- strain computed at preprocessing: epsilon = (R - R0) / (GF * R0), GF=5.64
- Binary label: 0=normal, 1=anomali

Create: data/generator.py

Class AircraftDataGenerator:
- 3 profiles: NarrowBodyShortHaul, WideBodyLongHaul, RepairedAircraft
  - differ in: strain amplitude, frequency, noise level
- generate_flight(profile, n_timesteps=500) → np.ndarray (n_timesteps, 1)
- inject_anomaly(flight_data, anomaly_type) → np.ndarray (n_timesteps, 1)
  - types: "crack" (spike), "overload" (sustained high), "drift" (gradual increase)
- generate_dataset(profile, n_flights=50, anomaly_rate=0.1)
  → list of ((n_timesteps, 1), label) tuples, label is 0 or 1

Use: numpy, scipy only.

__main__ block:
- Generate all 3 profiles, print shape + basic stats
- Save sample CSV: timestamp, strain, label → data/sample.csv

Next: output feeds directly into clients/fl_client.py as (n_timesteps, 1) tensor.
```

---

## PROMPT 2 — Module 2: FL Clients

```
Context:
- Input shape changed: (batch, timesteps, 1) — single rGO channel
- Binary classification: 0=normal, 1=anomali
- Loss: BCEWithLogitsLoss, output: 1 neuron (no Softmax)

Create: clients/model.py

Class StrainClassifier(nn.Module):
- Input: (batch, timesteps, 1)
- Architecture: 2x Conv1d → GlobalAvgPool → 2x Linear → output (1,)
- train_one_epoch(model, dataloader, optimizer, criterion) → avg loss
- evaluate(model, dataloader) → loss, accuracy, precision, recall

Create: clients/fl_client.py

Class CarbonClient(fl.client.NumPyClient):
- get_parameters(): return model weights as numpy arrays
- fit(parameters, config):
  1. load global parameters
  2. fine-tune local_epochs on local data (Personalized FedAvg)
  3. return updated parameters + metrics
- evaluate(parameters, config): return loss + accuracy

Helper: simulate_client(client_id, profile_name) → CarbonClient with generated data

Use: torch, flwr.

Next: flat weight vector from get_parameters() passed to
security/encryption.py encrypt_update().
```

---

## PROMPT 3 — Module 3: FL Server

```
You are building the federated learning server module for Carbon Sentinel.

Context:
- The server aggregates model updates from all aircraft clients
- We use standard FedAvg aggregation as the base
- The server must track baseline drift over time (trend monitoring for slow drift attack detection)
- Framework: Flower (flwr)

Create the file: server/fl_server.py

It should contain:

1. Class CarbonStrategy(fl.server.strategy.FedAvg):
   Override aggregate_fit() to:
   - Call parent FedAvg aggregation
   - After each round, compute the L2 norm of the change in global model weights vs previous round
   - Store this as drift_history (list of floats, one per round)
   - If drift exceeds a threshold (from config), log a WARNING: "Baseline drift detected in round X — possible slow drift attack"
   - Return aggregated result as normal

2. Function run_server(config) that:
   - Initializes the CarbonStrategy
   - Starts a Flower server with the strategy
   - After training completes, prints a drift history summary

3. Function get_drift_report(drift_history) that returns a dict with: max_drift, mean_drift, rounds_flagged (list of round numbers that exceeded threshold)

Use only flwr and numpy. Add comments explaining why drift monitoring catches slow poisoning attacks that FedAvg alone would miss.
```

---

## PROMPT 4 — Module 4: Security Layer

```
You are building the security module for Carbon Sentinel.

Context:
- We need to demonstrate two security features:
  1. FLTrust: detect and down-weight malicious model updates using cosine similarity against a trusted reference
  2. Byzantine attack simulation: one client sends a deliberately corrupted update to test FLTrust

Create two files:

--- FILE 1: security/fltrust.py ---
Implement FLTrust aggregation:

1. Function compute_trust_scores(client_updates, trusted_update):
   - client_updates: list of numpy arrays (one per client), each is a flattened model update vector
   - trusted_update: numpy array, the server's own update from its small trusted dataset
   - For each client update, compute cosine similarity with trusted_update
   - Apply ReLU (clip negative similarities to 0) — this is the trust score
   - Return list of trust scores (floats between 0 and 1)

2. Function fltrust_aggregate(client_updates, trust_scores):
   - Weighted average of client_updates using trust_scores as weights
   - Return aggregated update as numpy array

3. Function run_fltrust_demo(n_clients=5, n_byzantine=1):
   - Simulate n_clients honest updates (small random perturbations from a base vector)
   - Simulate n_byzantine malicious updates (large random vectors, far from base)
   - Compute trust scores
   - Print: each client's trust score, which clients were flagged (score < 0.3), aggregated result norm
   - This is the demo function — make it print clearly for presentation

--- FILE 2: security/attack_sim.py ---
Implement attack simulations for demo:

1. Function byzantine_attack(base_update, scale=10.0):
   - Returns a corrupted update: base_update * -scale (sign flip + amplify)
   - This simulates a compromised aircraft sending a maximally disruptive update

2. Function slow_drift_attack(base_update, drift_factor=0.05, round_number=1):
   - Returns a subtly poisoned update: base_update + small_bias * round_number
   - Simulates gradual poisoning that compounds over many rounds

3. Function simulate_attack_scenario(attack_type, n_rounds=10):
   - Runs a simulation showing how the attack evolves over rounds
   - Prints per-round stats showing drift accumulation
   - Returns drift_history list for plotting

Use only numpy. Add a __main__ block that runs both attack simulations and prints a clear summary showing why FLTrust catches byzantine but trend monitoring is needed for slow drift.
```

---

## PROMPT 5 — Module 5: Demo Dashboard

```
You are building the demo dashboard for Carbon Sentinel, a federated learning system for aircraft structural health monitoring. This dashboard will be shown at a competition, so it needs to look impressive and be easy to understand.

Create the file: demo/dashboard.py using Streamlit and Plotly.

The dashboard should have the following sections:

1. Header — Title "Carbon Sentinel" with a subtitle explaining what it does in one line

2. Sidebar — Controls:
   - Number of aircraft clients (slider, 3–10)
   - Number of FL rounds (slider, 5–20)
   - Anomaly injection toggle (checkbox)
   - Attack simulation toggle (checkbox, shows Byzantine attack scenario)
   - Run Simulation button

3. Section: Fleet Overview
   - Table showing each aircraft's profile (type, route, data points)
   - Color-code: green = normal, red = anomaly detected

4. Section: Training Progress
   - Line chart: global model accuracy per FL round
   - Line chart: per-aircraft local accuracy per round (multiple lines, one per aircraft)

5. Section: Anomaly Detection
   - If anomaly toggle is on: show which flights were flagged, with sensor heatmap (62 sensors × timesteps) using Plotly heatmap
   - Show True Positive / False Positive / False Negative counts

6. Section: Security Monitor (only if attack toggle is on)
   - Bar chart: trust scores per client (FLTrust result)
   - Red bar for any client with trust score < 0.3 (flagged as malicious)
   - Line chart: baseline drift over rounds

All charts should use Plotly. Data can be simulated/randomized for demo — it does not need to connect to the actual FL modules yet. Add st.spinner() during simulation. Make it visually clean with st.columns() layout.
```

---

## PROMPT 6 — Integration & Run Script

```
You are finalizing the Carbon Sentinel demo project. All modules have been built separately. Now create a single integration script that ties everything together for a local demo run without needing a real Flower server.

Create the file: run_demo.py

It should:

1. Load config from config.yaml

2. Simulate a full FL training loop locally (no actual Flower server needed):
   - Initialize a global StrainClassifier model
   - For each round:
     a. For each client (aircraft): run local fine-tuning (Personalized FedAvg), get updated weights
     b. Run FLTrust: compute trust scores, flag any malicious clients
     c. Aggregate honest updates using fltrust_aggregate()
     d. Update global model
     e. Compute and log drift

3. After training, print a final report:
   - Per-aircraft final accuracy
   - Rounds where drift was flagged
   - Any clients that were consistently low trust (possible Byzantine)

4. Save results to demo/results.json for the dashboard to load

5. Add a --attack flag (argparse) that injects one Byzantine client for the security demo

Make sure all imports work with the module structure defined in previous prompts. Add try/except with helpful error messages if a module is missing. The script should run end-to-end in under 2 minutes on a laptop CPU.
```

---

## Urutan Eksekusi

```
1. PROMPT 0  → setup repo & struktur folder
2. PROMPT 1  → data/generator.py
3. PROMPT 2  → clients/model.py + clients/fl_client.py
4. PROMPT 4  → security/fltrust.py + security/attack_sim.py  (bisa paralel sama PROMPT 3)
5. PROMPT 3  → server/fl_server.py
6. PROMPT 6  → run_demo.py  (integration)
7. PROMPT 5  → demo/dashboard.py  (terakhir, butuh results.json dari step 6)
```

---

## Tips Pake Prompt Ini

- Paste satu prompt per sesi baru supaya AI fokus
- Kalau ada error, paste error message-nya dan bilang: *"Fix this error, don't change the overall structure"*
- Kalau output kurang lengkap, bilang: *"Continue from where you stopped, output the rest of the file"*
- Setelah tiap modul selesai, test dulu dengan `python data/generator.py` sebelum lanjut ke modul berikutnya