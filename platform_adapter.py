"""
平台适配层 - 为 Android/iOS 和桌面提供统一的硬件接口
"""
import logging
import platform as _platform

logger = logging.getLogger(__name__)

PLATFORM = _platform.system().lower()
IS_MOBILE = False

try:
    from kivy.utils import platform as kivy_platform
    if kivy_platform in ("android", "ios"):
        IS_MOBILE = True
        PLATFORM = kivy_platform
except ImportError:
    pass

logger.info(f"Platform detected: {PLATFORM}, IS_MOBILE={IS_MOBILE}")


# ── 摄像头 ──────────────────────────────────────────────
class CameraAdapter:
    """跨平台摄像头适配器"""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._cap = None
        self._is_opened = False
        self._kivy_camera = None
        self._frame_buffer = None

    def open(self) -> bool:
        if IS_MOBILE and PLATFORM == "android":
            return self._open_android()
        return self._open_opencv()

    def _open_opencv(self) -> bool:
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._is_opened = True
            logger.info(f"OpenCV camera opened: {self.width}x{self.height}")
            return True
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return False

    def _open_android(self) -> bool:
        """Android 使用 OpenCV (通过 buildozer 集成)"""
        logger.info("Opening Android camera via OpenCV")
        return self._open_opencv()

    def read(self):
        if not self._is_opened:
            return False, None

        if self._cap is not None:
            ret, frame = self._cap.read()
            return ret, frame if ret else None

        return False, None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened


# ── 语音输出 ──────────────────────────────────────────────
class VoiceAdapter:
    """跨平台语音输出适配器"""

    def __init__(self, rate: int = 180, volume: float = 0.9):
        self.rate = rate
        self.volume = volume
        self._engine = None
        self._plyer_tts = None
        self._speak_lock = __import__('threading').Lock()
        self._init_engine()

    def _init_engine(self):
        if IS_MOBILE:
            try:
                from plyer import tts
                self._plyer_tts = tts
                logger.info("Using plyer TTS for mobile")
            except ImportError:
                logger.warning("plyer.tts not available")
        else:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
                voices = self._engine.getProperty("voices")
                for voice in voices:
                    if "chinese" in voice.name.lower() or "zh" in voice.id.lower():
                        self._engine.setProperty("voice", voice.id)
                        break
                logger.info("Using pyttsx3 TTS for desktop")
            except Exception as e:
                logger.warning(f"TTS init failed: {e}")

    def speak(self, text: str, priority: int = 3):
        if not text:
            return

        logger.info(f"[Voice] {text}")

        if self._plyer_tts:
            try:
                self._plyer_tts.speak(text)
            except Exception as e:
                logger.error(f"plyer TTS error: {e}")
        elif self._engine:
            import threading
            def _speak():
                with self._speak_lock:
                    try:
                        if priority <= 1:
                            self._engine.stop()
                        self._engine.say(text)
                        self._engine.runAndWait()
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
            threading.Thread(target=_speak, daemon=True).start()

    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass


# ── 震动输出 ──────────────────────────────────────────────
class VibrationAdapter:
    """跨平台震动输出适配器"""

    def __init__(self):
        self._plyer_vibrator = None
        self._serial_conn = None

        if IS_MOBILE:
            try:
                from plyer import vibrator
                self._plyer_vibrator = vibrator
                logger.info("Using plyer vibrator for mobile")
            except ImportError:
                logger.warning("plyer.vibrator not available")
        else:
            self._init_serial()

    def _init_serial(self):
        try:
            import serial
            from config import VIBRATION_SERIAL_PORT, VIBRATION_BAUD_RATE
            self._serial_conn = serial.Serial(
                VIBRATION_SERIAL_PORT, VIBRATION_BAUD_RATE, timeout=1
            )
            logger.info(f"Connected to vibration device")
        except Exception as e:
            logger.debug(f"Serial vibration not available: {e}")

    def vibrate(self, duration: float = 0.1):
        if self._plyer_vibrator:
            try:
                self._plyer_vibrator.vibrate(duration)
            except Exception as e:
                logger.error(f"Vibration error: {e}")
        elif self._serial_conn:
            try:
                self._serial_conn.write(f"both:{duration}\n".encode())
            except Exception as e:
                logger.error(f"Serial vibration error: {e}")

    def vibrate_direction(self, direction: str):
        patterns = {
            "left": 0.1,
            "right": 0.1,
            "stop": 0.5,
            "danger": 0.3,
            "approach": 0.2,
        }
        duration = patterns.get(direction, 0.1)
        self.vibrate(duration)

    def stop(self):
        if self._serial_conn:
            try:
                self._serial_conn.write(b"off:0\n")
                self._serial_conn.close()
            except Exception:
                pass


# ── 语音识别 ──────────────────────────────────────────────
class SpeechRecognizerAdapter:
    """跨平台语音识别适配器"""

    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self._recognizer = None
        self._running = False
        self._callback = None
        self._thread = None
        self._use_sounddevice = False
        self._init_recognizer()

    def _init_recognizer(self):
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            # Check if PyAudio works, fallback to sounddevice
            try:
                sr.Microphone()
            except Exception:
                try:
                    import sounddevice  # noqa: F401
                    self._use_sounddevice = True
                    logger.info("Speech recognizer initialized (sounddevice backend)")
                    return
                except ImportError:
                    pass
            logger.info("Speech recognizer initialized")
        except ImportError:
            logger.warning("speech_recognition not installed")

    def start_listening(self, callback):
        if not self._recognizer:
            logger.warning("Speech recognition not available")
            return

        self._callback = callback
        self._running = True

        import threading
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _record_audio_sd(self, duration=5, sample_rate=16000):
        """Record audio using sounddevice and return as AudioData."""
        import io, wave
        import sounddevice as sd
        import numpy as np
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                           channels=1, dtype='int16', blocking=True)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        buf.seek(0)
        import speech_recognition as sr
        with sr.AudioFile(buf) as source:
            return self._recognizer.record(source)

    def _listen_loop(self):
        import speech_recognition as sr

        if self._use_sounddevice:
            while self._running:
                try:
                    audio = self._record_audio_sd(duration=5)
                    text = self._recognizer.recognize_google(audio, language=self.language)
                    if text and self._callback:
                        self._callback(text)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                except Exception as e:
                    if not self._running:
                        break
                    logger.error(f"Speech recognition error: {e}")
            return

        try:
            mic = sr.Microphone()
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            return

        with mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)
            while self._running:
                try:
                    audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = self._recognizer.recognize_google(audio, language=self.language)
                    if text and self._callback:
                        self._callback(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception:
                    if not self._running:
                        break
                    continue

    def stop_listening(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
