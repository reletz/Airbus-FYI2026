"""Continuous acquisition loop for Carbon Sentinel MVP.

Reads strain data from Arduino via SensorDAQ and appends to live_stream.csv
in real-time so Streamlit dashboard can read the latest data.

Usage:
    uv run python acquire.py
    uv run python acquire.py --port /dev/ttyACM0 --baud 9600 --output data/live_stream.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from hardware.daq_reader import SensorDAQ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Stop signal received, finishing...")
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def acquire_continuous(
    port: str,
    baud: int,
    output: str,
    chunk_size: int,
    calibrate_samples: int,
) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV header if file doesn't exist yet
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    csv_file = open(out_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["timestamp", "R_ohm", "strain"])
        csv_file.flush()

    logger.info("Output: %s", out_path.resolve())

    daq = SensorDAQ(port=port, baud_rate=baud)

    logger.info("Calibrating with %d samples...", calibrate_samples)
    r0 = daq.calibrate(n_samples=calibrate_samples)
    logger.info("R0 = %.4f ohm — starting acquisition loop (Ctrl+C to stop)", r0)

    sample_index = 0

    try:
        while _running:
            buf = daq.read_live_stream(buffer_size=chunk_size)

            if buf.size == 0:
                logger.warning("Empty buffer, retrying in 1s...")
                time.sleep(1.0)
                continue

            strain = buf[:, 0]
            resistance = daq.R0 * (1.0 + daq.gauge_factor * strain)
            now = int(time.time() * 1000)  # ms epoch

            for i, (r, s) in enumerate(zip(resistance, strain)):
                writer.writerow([now + i, f"{r:.8f}", f"{s:.8e}"])

            csv_file.flush()
            sample_index += len(strain)
            logger.info("Wrote %d samples (total: %d)", len(strain), sample_index)

    finally:
        csv_file.close()
        logger.info("Acquisition stopped. Total samples written: %d", sample_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Arduino acquisition to CSV")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--output", default="data/live_stream.csv")
    parser.add_argument("--chunk", type=int, default=10, help="Samples per read cycle")
    parser.add_argument("--calibrate", type=int, default=20, help="Calibration samples")
    args = parser.parse_args()

    acquire_continuous(
        port=args.port,
        baud=args.baud,
        output=args.output,
        chunk_size=args.chunk,
        calibrate_samples=args.calibrate,
    )


if __name__ == "__main__":
    main()