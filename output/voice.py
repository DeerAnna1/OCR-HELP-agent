import logging
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TTS_RATE, TTS_VOLUME

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


class VoiceOutput:
    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME):
        self.rate = rate
        self.volume = volume
        self.engine = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        if not HAS_TTS:
            logger.warning("pyttsx3 not installed. Trying plyer TTS.")
            self._init_plyer_tts()
            return
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            voices = self.engine.getProperty("voices")
            for voice in voices:
                if "chinese" in voice.name.lower() or "zh" in voice.id.lower():
                    self.engine.setProperty("voice", voice.id)
                    break
        except Exception as e:
            logger.error(f"Failed to init TTS engine: {e}")
            self._init_plyer_tts()

    def _init_plyer_tts(self):
        try:
            from plyer import tts
            self._plyer_tts = tts
            logger.info("Using plyer TTS")
        except ImportError:
            self._plyer_tts = None
            logger.warning("No TTS engine available")

    def speak(self, text: str, priority: int = 3):
        if not text:
            return

        # 尝试使用 plyer (移动端)
        if hasattr(self, '_plyer_tts') and self._plyer_tts:
            try:
                self._plyer_tts.speak(text)
                return
            except Exception as e:
                logger.error(f"plyer TTS error: {e}")

        if self.engine is None:
            logger.info(f"[Voice] {text}")
            print(f"[Voice] {text}")
            return

        def _speak():
            with self._lock:
                try:
                    if priority <= 1:
                        self.engine.stop()
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS error: {e}")

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def stop(self):
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
