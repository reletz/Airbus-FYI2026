"""Single-channel DAQ reader for Carbon Sentinel MVP.

Reads Arduino UNO serial output and converts it to strain using:
epsilon = (R - R0) / (GF * R0)

The serial payload can be a raw resistance value, a raw ADC reading, or a
measured voltage. The reader keeps the downstream output as strain so the rest
of the FL pipeline does not need to change.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import serial
    from serial import SerialException
except ImportError as exc:
    raise ImportError(
        "pyserial is required for hardware.daq_reader. Install it with `pip install pyserial`."
    ) from exc


logger = logging.getLogger(__name__)


_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?(?:\d*)?)(?:[eE][-+]?\d+)?")


class SensorDAQ:
    """Read and preprocess single-channel rGO data from serial.

    Parameters
    ----------
    port:
        Serial device path, for example ``/dev/ttyACM0``.
    baud_rate:
        Serial baud rate used by the Arduino sketch.
    gauge_factor:
        Strain gauge factor used to convert resistance to strain.
    base_resistance:
        Baseline resistance ``R0`` in ohms.
    input_mode:
        ``"resistance"`` (default), ``"voltage"``, or ``"adc"``.
    vcc:
        Reference voltage used when ``input_mode`` is ``"voltage"`` or
        ``"adc"``.
    adc_resolution:
        ADC full-scale count used when ``input_mode`` is ``"adc"``.
    series_resistance:
        Known resistor value in the divider used by the Arduino sketch.
    serial_timeout:
        Read timeout in seconds.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 9600,
        gauge_factor: float = 5.64,
        base_resistance: float = 1000.0,
        input_mode: str = "resistance",
        vcc: float = 5.0,
        adc_resolution: int = 1023,
        series_resistance: float = 10000.0,
        serial_timeout: float = 1.0,
    ):
        self.port = port
        self.baud_rate = int(baud_rate)
        self.gauge_factor = float(gauge_factor)
        self.base_resistance = float(base_resistance)
        self.R0 = float(base_resistance)
        self.input_mode = input_mode.lower().strip()
        self.vcc = float(vcc)
        self.adc_resolution = int(adc_resolution)
        self.series_resistance = float(series_resistance)
        self.serial_timeout = float(serial_timeout)

        valid_modes = {"resistance", "voltage", "adc"}
        if self.input_mode not in valid_modes:
            raise ValueError(f"input_mode must be one of {sorted(valid_modes)}, got {input_mode!r}")

    def _extract_numeric_value(self, line: str) -> Optional[float]:
        """Extract a numeric payload from a serial line.

        Supports plain floats, CSV lines, and tagged payloads such as
        ``R=1234.5`` or ``adc:512``.
        """
        matches = _FLOAT_PATTERN.findall(line)
        if not matches:
            return None

        try:
            return float(matches[-1])
        except ValueError:
            return None

    def _voltage_to_resistance(self, voltage: float) -> Optional[float]:
        """Convert divider voltage to sensor resistance."""
        if voltage <= 0.0:
            return 0.0
        if voltage >= self.vcc:
            return None
        return self.series_resistance * (voltage / (self.vcc - voltage))

    def _adc_to_resistance(self, adc_value: float) -> Optional[float]:
        """Convert ADC reading to sensor resistance through divider voltage."""
        if adc_value <= 0.0:
            return 0.0
        if adc_value >= self.adc_resolution:
            return None
        voltage = (adc_value / float(self.adc_resolution)) * self.vcc
        return self._voltage_to_resistance(voltage)

    def _payload_to_resistance(self, payload: float) -> Optional[float]:
        """Normalize the incoming Arduino payload to resistance in ohms."""
        if self.input_mode == "resistance":
            return payload
        if self.input_mode == "voltage":
            return self._voltage_to_resistance(payload)
        return self._adc_to_resistance(payload)

    def _open_serial(self):
        return serial.Serial(self.port, self.baud_rate, timeout=self.serial_timeout)

    def _read_resistance(self, ser) -> Optional[float]:
        """Read one line and parse resistance float. Returns None on malformed input."""
        try:
            raw_line = ser.readline()
        except SerialException as exc:
            logger.exception("Serial read failure: %s", exc)
            return None

        if not raw_line:
            return None

        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            return None

        payload = self._extract_numeric_value(line)
        if payload is None:
            logger.warning("Non-float serial line ignored: %r", line)
            return None

        resistance = self._payload_to_resistance(payload)
        if resistance is None:
            logger.warning("Serial payload out of range for mode %s: %r", self.input_mode, line)
            return None

        return resistance

    def calibrate(self, n_samples: int = 100) -> float:
        """Calibrate baseline resistance R0 from no-load samples."""
        samples: List[float] = []

        try:
            with self._open_serial() as ser:
                attempts = 0
                max_attempts = max(n_samples * 20, 100)
                while len(samples) < n_samples and attempts < max_attempts:
                    attempts += 1
                    value = self._read_resistance(ser)
                    if value is None:
                        continue
                    samples.append(value)
        except SerialException as exc:
            logger.exception("SerialException during calibration on %s: %s", self.port, exc)
        except Exception as exc:
            logger.exception("Unexpected calibration error: %s", exc)

        if not samples:
            logger.warning("Calibration failed to read any valid sample, keeping default R0=%.6f", self.R0)
            return self.R0

        self.R0 = float(np.mean(samples))
        logger.info("Calibration complete with %d samples, R0=%.6f ohm", len(samples), self.R0)
        return self.R0

    def read_live_stream(self, buffer_size: int = 500) -> np.ndarray:
        """Read live resistance stream and return strain array of shape (N, 1)."""
        resistances: List[float] = []

        try:
            with self._open_serial() as ser:
                attempts = 0
                max_attempts = max(buffer_size * 20, 200)
                while len(resistances) < buffer_size and attempts < max_attempts:
                    attempts += 1
                    value = self._read_resistance(ser)
                    if value is None:
                        continue
                    resistances.append(value)
        except SerialException as exc:
            logger.exception("SerialException during live read on %s: %s", self.port, exc)
        except Exception as exc:
            logger.exception("Unexpected stream read error: %s", exc)

        if not resistances:
            return np.empty((0, 1), dtype=float)

        resistance_arr = np.asarray(resistances, dtype=float)
        strain = (resistance_arr - self.R0) / (self.gauge_factor * self.R0)
        return strain.reshape(-1, 1)

    def log_to_csv(self, buffer: np.ndarray, filepath: str) -> None:
        """Save timestamp, R_ohm, and strain columns to CSV."""
        arr = np.asarray(buffer, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"Expected buffer shape (n, 1), got {arr.shape}")

        strain = arr[:, 0]
        resistance = self.R0 * (1.0 + self.gauge_factor * strain)
        timestamps = np.arange(arr.shape[0], dtype=int)

        csv_data = np.column_stack([timestamps, resistance, strain])
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_path,
            csv_data,
            delimiter=",",
            header="timestamp,R_ohm,strain",
            comments="",
            fmt=["%d", "%.8f", "%.8e"],
        )


class DummySerial:
    """Dummy serial source for local demo without hardware."""

    def __init__(self, values: List[float]):
        self._lines = [f"{v:.6f}\n".encode("utf-8") for v in values]
        self._index = 0

    def readline(self):
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    base_r = 1000.0
    values = (base_r + rng.normal(0.0, 0.8, size=260)).tolist()

    reader = SensorDAQ(port="/dev/ttyUSB0", baud_rate=9600, base_resistance=base_r)

    original_serial = serial.Serial
    serial.Serial = lambda *args, **kwargs: DummySerial(values)  # type: ignore[assignment]
    try:
        reader.calibrate(n_samples=100)
        buffer = reader.read_live_stream(buffer_size=100)
        reader.log_to_csv(buffer, "data/live_stream.csv")
    finally:
        serial.Serial = original_serial  # type: ignore[assignment]

    print("Shape:", buffer.shape)
    print("First 3 rows:")
    print(buffer[:3])


if __name__ == "__main__":
    main()
