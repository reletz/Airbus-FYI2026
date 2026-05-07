import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import importlib.util

from typing import List


st.set_page_config(page_title="Carbon Sentinel", layout="wide")

# Initialize session state to persist simulation results across reruns
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None


def simulate_fleet(n_clients: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    profiles = ["NarrowBodyShortHaul", "WideBodyLongHaul", "RepairedAircraft"]
    fleet = []
    for i in range(n_clients):
        fleet.append(
            {
                "client_id": f"AC_{i+1:02d}",
                "profile": rng.choice(profiles),
                "route": f"R{rng.integers(100,999)}",
                "data_points": int(rng.integers(50, 200)),
            }
        )
    return pd.DataFrame(fleet)


def simulate_training(n_clients: int, n_rounds: int, attack=False, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Global accuracy
    base_acc = 0.6 + rng.normal(0, 0.02)
    global_acc = np.clip(base_acc + np.cumsum(rng.normal(0.01, 0.02, size=n_rounds)), 0, 1)

    # Per-client local accuracies
    local_acc = {}
    for i in range(n_clients):
        drift = rng.normal(0.0, 0.01, size=n_rounds).cumsum()
        local = np.clip(base_acc + rng.normal(0, 0.03) + drift + rng.normal(0, 0.01, size=n_rounds), 0, 1)
        local_acc[f"AC_{i+1:02d}"] = local

    # Trust scores and drift history
    trust_scores = np.clip(rng.normal(0.8, 0.1, size=n_clients), 0, 1)
    drift_history = np.abs(rng.normal(0.001, 0.002, size=n_rounds))

    if attack:
        # inject Byzantine client with low trust and cause drift in later rounds
        byz_idx = rng.integers(0, n_clients)
        trust_scores[byz_idx] = float(max(0.0, trust_scores[byz_idx] * 0.1))
        drift_history[-3:] += np.linspace(0.01, 0.05, 3)

    return global_acc, local_acc, trust_scores, drift_history


def load_demo_results(results_path: str = "demo/results.json"):
    path = Path(results_path)
    if not path.exists():
        return None
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_ifem_model():
    module_path = Path("math") / "ifem.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("ifem_local", str(module_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "ShapeSensing_iFEM"):
        return module.ShapeSensing_iFEM()
    return None


def load_flight_data_for_ifem() -> np.ndarray:
    """Load flight data for iFEM visualization. Returns (n_timesteps, n_sensors) array."""
    # Prefer live DAQ log, fallback to generated sample, then fallback random data.
    for candidate in [Path("data/live_stream.csv"), Path("data/sample.csv")]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if "strain" in df.columns:
                    single = df["strain"].to_numpy(dtype=float)
                    # Return as single-channel data (n_timesteps, 1)
                    return single.reshape(-1, 1)
            except Exception:
                continue
    # Fallback: generate random single-channel data
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 1e-3, size=(200, 1))


def plot_fleet_overview(df: pd.DataFrame, flagged: List[str] = None):
    df2 = df.copy()
    if flagged is None:
        df2["status"] = "Normal"
    else:
        df2["status"] = df2["client_id"].apply(lambda x: "Anomaly" if x in flagged else "Normal")
    st.write("**Fleet Overview**")
    styled = df2.style.apply(
        lambda row: ["background-color: #dcfce7" if row["status"] == "Normal" else "background-color: #fee2e2"] * len(row),
        axis=1,
    )
    st.dataframe(styled, use_container_width=True)


def main():
    st.title("Carbon Sentinel")
    st.markdown("Detect anomalies in aircraft strain sensor data with a privacy-preserving FL demo.")

    demo_results = load_demo_results()

    # Sidebar controls
    with st.sidebar:
        n_clients = st.slider("Number of aircraft clients", min_value=3, max_value=10, value=5)
        n_rounds = st.slider("Number of FL rounds", min_value=5, max_value=20, value=10)
        anomaly_toggle = st.checkbox("Inject anomalies (demo)", value=True)
        attack_toggle = st.checkbox("Simulate attack (Byzantine)", value=False)
        run_button = st.button("Run Simulation")

    # Initial static fleet table
    fleet_df = simulate_fleet(n_clients)

    # When Run button is clicked, store results in session state
    if run_button:
        with st.spinner("Running demo simulation..."):
            st.session_state.simulation_run = True
            if demo_results is not None:
                drift_history = demo_results.get("drift_history", [])
                trust_history = demo_results.get("trust_history", [])
                rounds = list(range(1, len(drift_history) + 1)) if drift_history else list(range(1, n_rounds + 1))
                trust_scores = trust_history[-1] if trust_history else np.clip(np.random.default_rng(42).normal(0.8, 0.1, size=n_clients), 0, 1)
                global_acc, local_acc, _, _ = simulate_training(n_clients, n_rounds, attack=attack_toggle)
                if "final_accuracies" in demo_results:
                    st.caption("Loaded saved demo results from demo/results.json")
            else:
                global_acc, local_acc, trust_scores, drift_history = simulate_training(
                    n_clients, n_rounds, attack=attack_toggle
                )
                rounds = list(range(1, n_rounds + 1))

            # Determine flagged clients (if anomaly toggle) randomly
            rng = np.random.default_rng(1)
            flagged_clients = []
            if anomaly_toggle:
                # flag 0-2 clients randomly
                k = rng.integers(0, min(3, n_clients) + 1)
                flagged_clients = [f"AC_{i+1:02d}" for i in rng.choice(n_clients, size=k, replace=False)]

            # Store results in session state for persistence
            st.session_state.sim_results = {
                "drift_history": drift_history,
                "trust_scores": trust_scores,
                "global_acc": global_acc,
                "local_acc": local_acc,
                "rounds": rounds,
                "flagged_clients": flagged_clients,
                "n_clients": n_clients,
                "n_rounds": n_rounds,
                "attack_toggle": attack_toggle,
                "anomaly_toggle": anomaly_toggle,
            }

    # Display results if simulation has been run (persists across slider changes)
    if st.session_state.simulation_run and st.session_state.sim_results is not None:
        sim = st.session_state.sim_results
        drift_history = sim["drift_history"]
        trust_scores = sim["trust_scores"]
        global_acc = sim["global_acc"]
        local_acc = sim["local_acc"]
        rounds = sim["rounds"]
        flagged_clients = sim["flagged_clients"]

        tab_main, tab_ifem = st.tabs(["FL Monitoring", "iFEM Shape Sensing"])

        with tab_main:
            # Fleet overview
            col1, col2 = st.columns([1, 2])
            with col1:
                plot_fleet_overview(fleet_df, flagged_clients)

            # Training progress charts
            acc_df = pd.DataFrame({"round": rounds, "global_acc": global_acc})
            fig_global = px.line(acc_df, x="round", y="global_acc", title="Global Accuracy per Round")
            with col2:
                st.plotly_chart(fig_global, use_container_width=True)

            # Per-client accuracies
            loc_long = []
            for cid, arr in local_acc.items():
                for r, v in enumerate(arr, start=1):
                    loc_long.append({"client_id": cid, "round": r, "acc": float(v)})
            loc_df = pd.DataFrame(loc_long)
            fig_local = px.line(loc_df, x="round", y="acc", color="client_id", title="Per-aircraft Local Accuracy")
            st.plotly_chart(fig_local, use_container_width=True)

            # Anomaly Detection section
            st.subheader("Anomaly Detection")
            if sim["anomaly_toggle"] and flagged_clients:
                st.markdown(f"Flagged flights/clients: {', '.join(flagged_clients)}")
                # show a sample heatmap for a flagged client
                sample = np.abs(np.random.normal(0, 1, size=(200, 62)))
                fig_heat = go.Figure(data=go.Heatmap(z=sample.T, colorscale="Viridis"))
                fig_heat.update_layout(title=f"Sensor heatmap (sample) — {flagged_clients[0]}", xaxis_title="timestep", yaxis_title="sensor")
                st.plotly_chart(fig_heat, use_container_width=True)

                # Simulated TP/FP/FN
                tp = int(len(flagged_clients) * 0.7)
                fp = int(len(flagged_clients) * 0.2)
                fn = max(0, len(flagged_clients) - tp - fp)
                metrics_df = pd.DataFrame([{"metric": "TP", "count": tp}, {"metric": "FP", "count": fp}, {"metric": "FN", "count": fn}])
                st.table(metrics_df)
            else:
                st.write("No anomalies injected in this run.")

            # Security Monitor
            if sim["attack_toggle"]:
                st.subheader("Security Monitor")
                trust_df = pd.DataFrame({"client_id": [f"AC_{i+1:02d}" for i in range(sim["n_clients"])], "trust": trust_scores})
                trust_df["flagged"] = trust_df["trust"] < 0.3
                fig_trust = px.bar(trust_df, x="client_id", y="trust", color=trust_df["flagged"], color_discrete_map={True: "red", False: "steelblue"}, title="Trust Scores per Client")
                st.plotly_chart(fig_trust, use_container_width=True)

                drift_df = pd.DataFrame({"round": rounds, "drift": drift_history})
                fig_drift = px.line(drift_df, x="round", y="drift", title="Baseline Drift over Rounds")
                st.plotly_chart(fig_drift, use_container_width=True)

            if demo_results is not None:
                norm_plain = demo_results.get("plaintext_norm_history", [])
                norm_enc = demo_results.get("encrypted_norm_history", [])
                if norm_plain and norm_enc:
                    norm_rounds = list(range(1, min(len(norm_plain), len(norm_enc)) + 1))
                    norm_df = pd.DataFrame(
                        {
                            "round": norm_rounds,
                            "plaintext_norm": norm_plain[: len(norm_rounds)],
                            "encrypted_norm": norm_enc[: len(norm_rounds)],
                        }
                    )
                    fig_norm = px.line(
                        norm_df,
                        x="round",
                        y=["plaintext_norm", "encrypted_norm"],
                        title="Plaintext vs Encrypted Update Norms",
                        labels={"value": "norm", "variable": "series"},
                    )
                    st.plotly_chart(fig_norm, use_container_width=True)

        with tab_ifem:
            st.subheader("iFEM Shape Sensing")
            ifem = load_ifem_model()
            if ifem is None:
                st.warning("iFEM module not available at math/ifem.py")
            else:
                flight_data = load_flight_data_for_ifem()
                timestep = st.slider("Select timestep", min_value=0, max_value=max(0, flight_data.shape[0] - 1), value=0)
                strain_at_timestep = flight_data[timestep]
                displacement = ifem.reconstruct_displacement(strain_at_timestep)

                surface = go.Figure(
                    data=[go.Surface(z=displacement, colorscale="RdBu")]
                )
                surface.update_layout(
                    title="iFEM Reconstructed Displacement",
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Displacement"),
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(surface, use_container_width=True)
                st.caption(f"Reconstructed wing displacement field at timestep {timestep}")

        st.success("Simulation complete")

    else:
        st.info("Adjust controls in the sidebar and click 'Run Simulation' to start")


if __name__ == "__main__":
    main()
