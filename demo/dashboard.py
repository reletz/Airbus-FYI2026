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
# iFEM playback state
if "ifem_autoplay" not in st.session_state:
    st.session_state.ifem_autoplay = False
if "ifem_slider" not in st.session_state:
    st.session_state.ifem_slider = 0

# Fixed playback frame rate for iFEM "video"
IFEM_FPS = 24

# Accent works on both light and dark backgrounds
ACCENT = "#0ea5e9"


def get_palette():
    """Return a color palette matching the user's active (light/dark) theme."""
    is_dark = True
    try:
        ttype = getattr(st.context.theme, "type", None)
        if ttype in ("light", "dark"):
            is_dark = ttype == "dark"
        else:
            raise ValueError
    except Exception:
        try:
            base = st.get_option("theme.base")
            if base in ("light", "dark"):
                is_dark = base == "dark"
        except Exception:
            pass

    if is_dark:
        return {
            "is_dark": True,
            "text": "#f1f5f9",
            "body": "#cbd5e1",
            "muted": "#94a3b8",
            "grid": "rgba(148,163,184,0.12)",
            "zero": "rgba(148,163,184,0.2)",
            "card_bg": "rgba(255,255,255,0.03)",
            "border": "rgba(148,163,184,0.16)",
            "row_border": "rgba(148,163,184,0.08)",
            "ok": "#4ade80", "bad": "#f87171",
            "ok_pill_bg": "rgba(74,222,128,.16)", "ok_pill_fg": "#86efac",
            "bad_pill_bg": "rgba(248,113,113,.16)", "bad_pill_fg": "#fca5a5",
            "fill": "rgba(14,165,233,0.10)",
            "colorway": ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
                         "#fb7185", "#22d3ee", "#facc15", "#4ade80", "#f87171"],
        }
    return {
        "is_dark": False,
        "text": "#0f172a",
        "body": "#334155",
        "muted": "#64748b",
        "grid": "rgba(15,23,42,0.08)",
        "zero": "rgba(15,23,42,0.16)",
        "card_bg": "rgba(15,23,42,0.02)",
        "border": "rgba(15,23,42,0.12)",
        "row_border": "rgba(15,23,42,0.06)",
        "ok": "#16a34a", "bad": "#dc2626",
        "ok_pill_bg": "rgba(22,163,74,.12)", "ok_pill_fg": "#15803d",
        "bad_pill_bg": "rgba(220,38,38,.12)", "bad_pill_fg": "#b91c1c",
        "fill": "rgba(14,165,233,0.12)",
        "colorway": ["#0284c7", "#059669", "#d97706", "#db2777", "#7c3aed",
                     "#e11d48", "#0891b2", "#ca8a04", "#16a34a", "#dc2626"],
    }


def inject_global_css(pal):
    css = """
    <style>
    [data-testid="stMetric"] {
        background: __CARD_BG__;
        border: 1px solid __BORDER__;
        border-radius: 14px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] p { color: __MUTED__; font-size: 13px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .cs-section { color: __TEXT__; font-size: 18px; font-weight: 600; margin: 6px 0 10px; }
    </style>
    """
    css = (css.replace("__CARD_BG__", pal["card_bg"])
              .replace("__BORDER__", pal["border"])
              .replace("__MUTED__", pal["muted"])
              .replace("__TEXT__", pal["text"]))
    st.markdown(css, unsafe_allow_html=True)


def style_fig(fig, pal, height=None):
    """Apply a consistent, theme-aware transparent theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["body"], family="Inter, system-ui, sans-serif", size=13),
        title=dict(font=dict(size=17, color=pal["text"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        colorway=pal["colorway"],
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor=pal["grid"], zerolinecolor=pal["zero"])
    fig.update_yaxes(gridcolor=pal["grid"], zerolinecolor=pal["zero"])
    if height:
        fig.update_layout(height=height)
    return fig


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


def render_fleet_overview(df: pd.DataFrame, flagged: List[str], pal):
    """Render the fleet as a styled, theme-aware HTML table with status pills."""
    flagged = set(flagged or [])

    th = (f"padding:10px 14px;color:{pal['muted']};font-size:11px;letter-spacing:.06em;"
          "text-transform:uppercase;font-weight:600;"
          f"border-bottom:1px solid {pal['border']}")
    header = (
        "<tr>"
        f"<th style='text-align:left;{th}'>Aircraft</th>"
        f"<th style='text-align:left;{th}'>Profile</th>"
        f"<th style='text-align:left;{th}'>Route</th>"
        f"<th style='text-align:right;{th}'>Data points</th>"
        f"<th style='text-align:center;{th}'>Status</th>"
        "</tr>"
    )

    body = ""
    for _, r in df.iterrows():
        anom = r["client_id"] in flagged
        accent = pal["bad"] if anom else pal["ok"]
        pill_bg = pal["bad_pill_bg"] if anom else pal["ok_pill_bg"]
        pill_fg = pal["bad_pill_fg"] if anom else pal["ok_pill_fg"]
        label = "Anomaly" if anom else "Normal"
        body += (
            f"<tr style='border-bottom:1px solid {pal['row_border']}'>"
            f"<td style='padding:11px 14px;border-left:3px solid {accent};font-family:ui-monospace,monospace;font-weight:600;color:{pal['text']};font-size:13px'>{r['client_id']}</td>"
            f"<td style='padding:11px 14px;color:{pal['body']};font-size:13px'>{r['profile']}</td>"
            f"<td style='padding:11px 14px;color:{pal['muted']};font-size:13px;font-family:ui-monospace,monospace'>{r['route']}</td>"
            f"<td style='padding:11px 14px;color:{pal['body']};font-size:13px;text-align:right'>{r['data_points']}</td>"
            f"<td style='padding:11px 14px;text-align:center'><span style='background:{pill_bg};color:{pill_fg};padding:3px 11px;border-radius:999px;font-size:11px;font-weight:600'>{label}</span></td>"
            "</tr>"
        )

    html = (
        f"<div style='border:1px solid {pal['border']};border-radius:14px;"
        f"overflow:hidden;background:{pal['card_bg']}'>"
        "<table style='width:100%;border-collapse:collapse'>"
        f"<thead>{header}</thead><tbody>{body}</tbody></table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def main():
    pal = get_palette()

    st.title("Carbon Sentinel")
    st.markdown("Detect anomalies in aircraft strain sensor data with a privacy-preserving FL demo.")
    inject_global_css(pal)

    demo_results = load_demo_results()

    # Sidebar controls (each carries a help tooltip explaining what it does)
    with st.sidebar:
        st.markdown("### Controls")
        n_clients = st.slider(
            "Number of aircraft clients", min_value=3, max_value=10, value=5,
            help="How many aircraft act as federated-learning clients. Each one trains "
                 "a model locally on its own strain data and shares only model updates, "
                 "never the raw data.",
        )
        n_rounds = st.slider(
            "Number of FL rounds", min_value=5, max_value=20, value=10,
            help="Number of global training rounds. In each round every client trains "
                 "locally, then the server aggregates their updates into one global model.",
        )
        anomaly_toggle = st.checkbox(
            "Inject anomalies (demo)", value=True,
            help="Randomly mark a few aircraft as showing abnormal strain patterns so you "
                 "can see the anomaly-detection view in action.",
        )
        attack_toggle = st.checkbox(
            "Simulate attack (Byzantine)", value=False,
            help="Simulate a malicious client that sends poisoned model updates, to "
                 "demonstrate the trust / security monitoring.",
        )
        run_button = st.button(
            "Run Simulation",
            help="Run the simulated federated training and refresh all charts below.",
        )

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
            with st.expander("ℹ️ About this tab"):
                st.write(
                    "This tab monitors a **federated-learning (FL)** system: each aircraft "
                    "trains an anomaly-detection model locally and shares only encrypted "
                    "model updates — raw sensor data never leaves the aircraft. Below you'll "
                    "find the fleet roster, how the shared model learns over time, which "
                    "aircraft look anomalous, and (if enabled) how the system defends against "
                    "a malicious client."
                )

            # KPI cards
            acc_first = float(global_acc[0])
            acc_last = float(global_acc[-1])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Aircraft in fleet", sim["n_clients"],
                      help="Total aircraft taking part as FL clients in this run.")
            k2.metric("Anomalies flagged", len(flagged_clients),
                      help="How many aircraft were flagged with abnormal strain behaviour.")
            k3.metric(
                "Final global accuracy", f"{acc_last * 100:.1f}%",
                delta=f"{(acc_last - acc_first) * 100:+.1f} pts",
                help="Accuracy of the shared global model after the final round. "
                     "The delta is the change since round 1.",
            )
            k4.metric("Avg client trust", f"{float(np.mean(trust_scores)):.2f}",
                      help="Mean trust score across clients (1.0 = fully trusted). "
                           "A low value suggests a misbehaving client.")

            st.divider()

            # Fleet overview + global accuracy side by side
            col1, col2 = st.columns([1, 1.35])
            with col1:
                st.markdown("<div class='cs-section'>Fleet Overview</div>", unsafe_allow_html=True)
                render_fleet_overview(fleet_df, flagged_clients, pal)
                st.caption("Each row is one aircraft / FL client. The green or red pill and "
                           "left bar mark whether its strain behaviour is normal or flagged.")
            with col2:
                acc_df = pd.DataFrame({"round": rounds, "global_acc": global_acc})
                fig_global = px.line(acc_df, x="round", y="global_acc", title="Global Accuracy per Round")
                fig_global.update_traces(
                    mode="lines+markers", line=dict(width=3, color=ACCENT),
                    marker=dict(size=7), fill="tozeroy", fillcolor=pal["fill"],
                )
                fig_global.update_yaxes(title_text="accuracy")
                st.plotly_chart(style_fig(fig_global, pal, height=340), use_container_width=True)
                st.caption("Accuracy of the aggregated global model as it improves across "
                           "training rounds. Upward trend = the fleet is learning together.")

            # Per-client accuracies
            loc_long = []
            for cid, arr in local_acc.items():
                for r, v in enumerate(arr, start=1):
                    loc_long.append({"client_id": cid, "round": r, "acc": float(v)})
            loc_df = pd.DataFrame(loc_long)
            fig_local = px.line(loc_df, x="round", y="acc", color="client_id", title="Per-aircraft Local Accuracy")
            fig_local.update_traces(line=dict(width=2))
            st.plotly_chart(style_fig(fig_local, pal, height=380), use_container_width=True)
            st.caption("One line per aircraft showing its local model's accuracy. A line that "
                       "diverges from the pack can hint at sensor drift or bad local data.")

            # Anomaly Detection section
            st.markdown("<div class='cs-section'>Anomaly Detection</div>", unsafe_allow_html=True)
            if sim["anomaly_toggle"] and flagged_clients:
                st.markdown(f"Flagged flights/clients: **{', '.join(flagged_clients)}**")
                # show a sample heatmap for a flagged client
                sample = np.abs(np.random.normal(0, 1, size=(200, 62)))
                fig_heat = go.Figure(data=go.Heatmap(z=sample.T, colorscale="Viridis", colorbar=dict(outlinewidth=0)))
                fig_heat.update_layout(title=f"Sensor heatmap (sample) — {flagged_clients[0]}", xaxis_title="timestep", yaxis_title="sensor")
                st.plotly_chart(style_fig(fig_heat, pal, height=360), use_container_width=True)
                st.caption("Strain magnitude for each sensor (rows) over time (columns) on a "
                           "flagged aircraft. Bright bands are moments / sensors with unusual strain.")

                # Simulated TP/FP/FN
                tp = int(len(flagged_clients) * 0.7)
                fp = int(len(flagged_clients) * 0.2)
                fn = max(0, len(flagged_clients) - tp - fp)
                m1, m2, m3 = st.columns(3)
                m1.metric("True positives", tp, help="Real anomalies the detector caught correctly.")
                m2.metric("False positives", fp, help="Normal aircraft wrongly flagged (false alarms).")
                m3.metric("False negatives", fn, help="Real anomalies the detector missed.")
            else:
                st.info("No anomalies injected in this run. Enable 'Inject anomalies' in the "
                        "sidebar and re-run to see the detection view.")

            # Security Monitor
            if sim["attack_toggle"]:
                st.markdown("<div class='cs-section'>Security Monitor</div>", unsafe_allow_html=True)
                st.caption("Active because 'Simulate attack' is on — these views show how the "
                           "server detects and contains a malicious (Byzantine) client.")
                trust_df = pd.DataFrame({"client_id": [f"AC_{i+1:02d}" for i in range(sim["n_clients"])], "trust": trust_scores})
                trust_df["flagged"] = trust_df["trust"] < 0.3
                fig_trust = px.bar(
                    trust_df, x="client_id", y="trust", color=trust_df["flagged"],
                    color_discrete_map={True: pal["bad"], False: ACCENT},
                    title="Trust Scores per Client",
                )
                fig_trust.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig_trust, pal, height=320), use_container_width=True)
                st.caption("Server-estimated trust per client. Bars under 0.3 (red) are treated "
                           "as likely malicious and down-weighted during aggregation.")

                drift_df = pd.DataFrame({"round": rounds, "drift": drift_history})
                fig_drift = px.line(drift_df, x="round", y="drift", title="Baseline Drift over Rounds")
                fig_drift.update_traces(line=dict(width=3, color="#f59e0b"))
                st.plotly_chart(style_fig(fig_drift, pal, height=320), use_container_width=True)
                st.caption("How far the model baseline shifts each round. A spike in later "
                           "rounds often marks the moment a poisoning attack takes effect.")

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
                    st.plotly_chart(style_fig(fig_norm, pal, height=320), use_container_width=True)
                    st.caption("Size (norm) of model updates before vs after encryption. The two "
                               "lines tracking closely shows privacy protection doesn't distort learning.")

        with tab_ifem:
            st.markdown("<div class='cs-section'>iFEM Shape Sensing</div>", unsafe_allow_html=True)
            with st.expander("ℹ️ About this tab"):
                st.write(
                    "**iFEM** (inverse Finite Element Method) reconstructs the wing's full 3D "
                    "deformation purely from a handful of discrete strain-sensor readings — no "
                    "displacement sensors required. Press **Play** to animate the reconstructed "
                    "shape across the flight (24 fps), or drag the **timeline** to inspect any "
                    "single moment. **Reset** returns to the first timestep."
                )
            ifem = load_ifem_model()
            if ifem is None:
                st.warning("iFEM module not available at math/ifem.py")
            else:
                flight_data = load_flight_data_for_ifem()
                max_timestep = max(0, flight_data.shape[0] - 1)

                # Keep the slider state inside the valid range (data may have changed).
                if st.session_state.ifem_slider > max_timestep:
                    st.session_state.ifem_slider = 0

                # ---- Controls (parent scope -> reliable full reruns) ----
                col_play, col_reset = st.columns(2)
                with col_play:
                    play_label = "⏸ Pause" if st.session_state.ifem_autoplay else "▶ Play"
                    if st.button(play_label, key="ifem_play_btn", use_container_width=True,
                                 help="Animate the reconstructed wing shape over time, like a video."):
                        st.session_state.ifem_autoplay = not st.session_state.ifem_autoplay
                with col_reset:
                    if st.button("🔄 Reset", key="ifem_reset_btn", use_container_width=True,
                                 help="Stop playback and jump back to the first timestep."):
                        st.session_state.ifem_autoplay = False
                        st.session_state.ifem_slider = 0

                st.caption("Drag the timeline to scrub to a specific moment (pause first for "
                           "precise control).")

                # Arm the timer only while playing; fixed at IFEM_FPS.
                run_every = (1.0 / IFEM_FPS) if st.session_state.ifem_autoplay else None

                # Defined fresh each parent run so run_every reflects the current state.
                @st.fragment(run_every=run_every)
                def render_ifem_frame():
                    # Advance the frame BEFORE the slider widget is created (only legal
                    # place to write a widget-keyed session value).
                    if st.session_state.ifem_autoplay:
                        nxt = st.session_state.ifem_slider + 1
                        st.session_state.ifem_slider = 0 if nxt > max_timestep else nxt  # loop

                    timestep = st.slider(
                        "Timestep",
                        min_value=0,
                        max_value=max_timestep,
                        label_visibility="collapsed",
                        key="ifem_slider",
                    )

                    strain_at_timestep = flight_data[timestep]
                    displacement = ifem.reconstruct_displacement(strain_at_timestep)

                    surface = go.Figure(data=[go.Surface(z=displacement, colorscale="RdBu")])
                    surface.update_layout(
                        title="iFEM Reconstructed Displacement",
                        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Displacement"),
                        margin=dict(l=0, r=0, b=0, t=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=pal["body"]),
                        uirevision="ifem",  # keep camera angle stable between frames
                    )
                    st.plotly_chart(surface, use_container_width=True)
                    st.caption(
                        f"Reconstructed wing displacement field at timestep {timestep} / "
                        f"{max_timestep}. Height/color = how far that part of the wing has "
                        "deflected from rest, inferred from the strain sensors."
                    )

                render_ifem_frame()

        st.success("Simulation complete")

    else:
        st.info("Adjust controls in the sidebar and click 'Run Simulation' to start")


if __name__ == "__main__":
    main()