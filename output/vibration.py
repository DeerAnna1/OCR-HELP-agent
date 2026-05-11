import logging
import time
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIBRATION_ENABLED, VIBRATION_SERIAL_PORT, VIBRATION_BAUD_RATE

logger = logging.getLogger(__name__)

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from plyer import vibrator as plyer_vibrator
    HAS_PLYER_VIBRATOR = True
except ImportError:
    HAS_PLYER_VIBRATOR = False


class VibrationPattern:
    LEFT_SHORT = "left_short"
    RIGHT_SHORT = "right_short"
    BOTH_URGENT = "both_urgent"
    SLOW_CONTINUOUS = "slow_continuous"
    FAST_CONTINUOUS = "fast_continuous"
    SINGLE_PULSE = "single_pulse"
    DOUBLE_PULSE = "double_pulse"


PATTERN_DURATIONS = {
    VibrationPattern.LEFT_SHORT: [(0.1, "left")],
    VibrationPattern.RIGHT_SHORT: [(0.1, "right")],
    VibrationPattern.BOTH_URGENT: [(0.1, "both"), (0.05, "off"), (0.1, "both"), (0.05, "off"), (0.1, "both")],
    VibrationPattern.SLOW_CONTINUOUS: [(0.3, "both"), (0.3, "off")] * 3,
    VibrationPattern.FAST_CONTINUOUS: [(0.1, "both"), (0.05, "off")] * 5,
    VibrationPattern.SINGLE_PULSE: [(0.2, "both")],
    VibrationPattern.DOUBLE_PULSE: [(0.15, "both"), (0.1, "off"), (0.15, "both")],
}


class VibrationOutput:
    def __init__(self, port: str = VIBRATION_SERIAL_PORT, baud_rate: int = VIBRATION_BAUD_RATE):
        self.enabled = VIBRATION_ENABLED
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self._lock = threading.Lock()

        if HAS_PLYER_VIBRATOR:
            logger.info("Using plyer vibrator (mobile)")
            return

        if self.enabled:
            self._connect()

    def _connect(self):
        if not HAS_SERIAL:
            logger.warning("pyserial not installed. Vibration output disabled.")
            self.enabled = False
            return
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            logger.info(f"Connected to vibration device on {self.port}")
        except Exception as e:
            logger.warning(f"Failed to connect vibration device: {e}")
            self.serial_conn = None

    def vibrate(self, pattern: str, direction: str = "both"):
        if HAS_PLYER_VIBRATOR:
            duration = self._pattern_to_duration(pattern)
            try:
                plyer_vibrator.vibrate(duration)
            except Exception as e:
                logger.error(f"plyer vibration error: {e}")
            return

        if not self.enabled:
            return
        commands = PATTERN_DURATIONS.get(pattern, [])
        if not commands:
            return

        def _execute():
            with self._lock:
                for duration, motor in commands:
                    if motor == "off":
                        time.sleep(duration)
                        continue
                    self._send_command(motor, duration)
                    time.sleep(duration)
                self._send_command("off", 0)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()

    def _pattern_to_duration(self, pattern: str) -> float:
        patterns = {
            VibrationPattern.LEFT_SHORT: 0.1,
            VibrationPattern.RIGHT_SHORT: 0.1,
            VibrationPattern.BOTH_URGENT: 0.3,
            VibrationPattern.SLOW_CONTINUOUS: 0.5,
            VibrationPattern.FAST_CONTINUOUS: 0.3,
            VibrationPattern.SINGLE_PULSE: 0.2,
            VibrationPattern.DOUBLE_PULSE: 0.3,
        }
        return patterns.get(pattern, 0.1)

    def vibrate_direction(self, direction: str):
        pattern_map = {
            "left": VibrationPattern.LEFT_SHORT,
            "right": VibrationPattern.RIGHT_SHORT,
            "stop": VibrationPattern.BOTH_URGENT,
            "approach": VibrationPattern.SLOW_CONTINUOUS,
            "danger": VibrationPattern.FAST_CONTINUOUS,
        }
        pattern = pattern_map.get(direction, VibrationPattern.SINGLE_PULSE)
        self.vibrate(pattern)

    def _send_command(self, motor: str, duration: float):
        if self.serial_conn is None:
            logger.debug(f"[Vibration] {motor} for {duration}s")
            return
        try:
            cmd = f"{motor}:{duration}\n"
            self.serial_conn.write(cmd.encode())
        except Exception as e:
            logger.error(f"Vibration command error: {e}")

    def stop(self):
        if self.serial_conn:
            try:
                self._send_command("off", 0)
                self.serial_conn.close()
            except Exception:
                pass
