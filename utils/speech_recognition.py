import logging
import threading
import io
import wave
import struct

logger = logging.getLogger(__name__)

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import sounddevice as sd
    import numpy as np
    HAS_SD = True
except ImportError:
    HAS_SD = False


class SpeechRecognizer:
    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self.recognizer = None
        self._running = False
        self._callback = None
        self._thread = None
        self._use_sounddevice = False
        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        # Detect if PyAudio is available
        if HAS_SR:
            try:
                sr.Microphone()
            except Exception:
                if HAS_SD:
                    self._use_sounddevice = True
                    logger.info("PyAudio not available, using sounddevice for microphone")

    def start_listening(self, callback):
        if not HAS_SR:
            logger.warning("SpeechRecognition not installed. Voice input disabled.")
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _record_audio_sd(self, duration=5, sample_rate=16000):
        """Record audio using sounddevice and return as AudioData."""
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                           channels=1, dtype='int16', blocking=True)
        # Convert to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        buf.seek(0)
        with sr.AudioFile(buf) as source:
            return self.recognizer.record(source)

    def _listen_loop(self):
        if self.recognizer is None:
            return

        if self._use_sounddevice:
            self._listen_loop_sd()
            return

        mic = None
        try:
            mic = sr.Microphone()
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            return

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self._running:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    if text and self._callback:
                        self._callback(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                except Exception as e:
                    logger.error(f"Speech recognition error: {e}")
                    if not self._running:
                        break

    def _listen_loop_sd(self):
        """Listen loop using sounddevice backend."""
        while self._running:
            try:
                audio = self._record_audio_sd(duration=5)
                text = self.recognizer.recognize_google(audio, language=self.language)
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

    def stop_listening(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
