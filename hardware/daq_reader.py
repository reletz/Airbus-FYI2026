"""Live sensor acquisition for Carbon Sentinel.

This module reads comma-separated rGO sensor values from a serial port and
converts resistance deltas to strain using the gauge factor formula.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

try:
    import serial
    from serial import SerialException
except ImportError as exc:  # pragma: no cover - dependency error is surfaced clearly
    raise ImportError(
        "pyserial is required for hardware.daq_reader. Install it with `pip install pyserial`."
    ) from exc


logger = logging.getLogger(__name__)


class SensorDAQ:
    """Read live strain measurements from a serial-connected sensor array."""

    SENSOR_COUNT = 62

    def __init__(self, port: str, baud_rate: int, gauge_factor: float = 5.64, base_resistance: float = 1000.0):
        self.port = port
        self.baud_rate = baud_rate
        self.gauge_factor = float(gauge_factor)
        self.base_resistance = float(base_resistance)

    def _line_to_strain(self, line: str) -> Optional[np.ndarray]:
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) != self.SENSOR_COUNT:
            logger.warning(
                "Malformed serial line with %d values instead of %d: %r",
                len(parts),
                self.SENSOR_COUNT,
                line,
            )
            return None

        try:
            resistances = np.asarray([float(value) for value in parts], dtype=float)
        except ValueError:
            logger.warning("Malformed numeric values in serial line: %r", line)
            return None

        delta_r = resistances - self.base_resistance
        strain = (delta_r / self.base_resistance) / self.gauge_factor
        return strain

    def read_live_stream(self, buffer_size: int = 500) -> np.ndarray:
        """Read live serial data and return a buffer of strain rows.

        Malformed lines and serial errors are logged and skipped. The returned
        array contains as many valid rows as were read before the buffer filled
        or the connection failed.
        """
        collected: List[np.ndarray] = []

        try:
            with serial.Serial(self.port, self.baud_rate, timeout=1) as ser:
                while len(collected) < buffer_size:
                    try:
                        raw_line = ser.readline()
                    except SerialException as exc:
                        logger.exception("Serial read failed: %s", exc)
                        break

                    if not raw_line:
                        continue

                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        logger.warning("Could not decode serial line: %r", raw_line)
                        continue

                    if not line:
                        continue

                    strain = self._line_to_strain(line)
                    if strain is None:
                        continue
                    collected.append(strain)
        except SerialException as exc:
            logger.exception("Could not open serial port %s at %d baud: %s", self.port, self.baud_rate, exc)
        except Exception as exc:
            logger.exception("Unexpected DAQ error: %s", exc)

        if not collected:
            return np.empty((0, self.SENSOR_COUNT), dtype=float)
        return np.vstack(collected)


class DummySerial:
    """Minimal serial stand-in for the module demo."""

    def __init__(self, lines: List[str]):
        self._lines = [line.encode("utf-8") for line in lines]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_lines = []
    base = 1000.0
    for row in range(5):
        values = base + np.linspace(0, 5, SensorDAQ.SENSOR_COUNT) + row * 0.5
        demo_lines.append(",".join(f"{value:.3f}" for value in values))

    dummy_serial = DummySerial(demo_lines)
    reader = SensorDAQ(port="/dev/ttyUSB0", baud_rate=115200)

    original_serial = serial.Serial
    serial.Serial = lambda *args, **kwargs: dummy_serial  # type: ignore[assignment]
    try:
        data = reader.read_live_stream(buffer_size=5)
    finally:
        serial.Serial = original_serial  # type: ignore[assignment]

    print("Shape:", data.shape)
    print("First 3 rows:")
    print(data[:3])
