# Carbon Sentinel

> Federated Learning for aircraft structural health monitoring — privacy-preserving, attack-resilient anomaly detection on strain sensor data.

What it does
---------------
Carbon Sentinel simulates and demonstrates a federated learning (FL) pipeline that detects anomalies in aircraft strain sensor readings. The project is intended as a research demo showcasing privacy-preserving training across multiple aircraft (clients) while providing defenses against malicious or drifting updates.

Three pillars
---------------
- rGO sensors: Reduced Graphene Oxide (rGO) sensor concept for dense strain sensing across aircraft structures — provides the raw high-dimensional strain signals used in demos.
- iFEM math: Inverse Finite Element Method (iFEM) modeling to convert sensor measurements into meaningful structural strain fields and features used by the learning pipeline.
- FL system: A Flower (flwr) based federated learning system that trains a central model while clients keep raw data locally; includes security layers (FLTrust, drift monitoring) to resist poisoning and slow-drift attacks.

Quick install
---------------
1. Create a virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

How to run (demo)
---------------
- A lightweight local demo runner is provided as `run_demo.py` which simulates clients and runs a federated training loop without an actual Flower server.
- For an interactive dashboard (presentation), use Streamlit:

   streamlit run demo/dashboard.py

See the project structure for modules: `data/`, `clients/`, `server/`, `security/`, and `demo/`.

License & notes
---------------
This scaffold is a demo/research artifact. Implementations of sensors, iFEM math, and production-secure FL should be adapted and validated before any real-world use.