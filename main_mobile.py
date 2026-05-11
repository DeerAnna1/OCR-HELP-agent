"""
GuideVision Mobile - 视觉辅助 Agent 手机端应用
自动模式：AI 自主识别场景，语音交互驱动
"""
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import platform
from kivy.graphics.texture import Texture

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GuideVisionMobile")

# 请求 Android 权限
if platform == "android":
    try:
        from android.permissions import request_permissions, Permission
        request_permissions([
            Permission.CAMERA,
            Permission.RECORD_AUDIO,
            Permission.VIBRATE,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE,
        ])
    except Exception as e:
        logger.warning(f"Permission request failed: {e}")

# OpenCV 可选
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available, using Kivy camera")


from config import WorkMode
from agent.state_machine import AgentStateMachine


class GuideVisionMobileLayout(FloatLayout):
    """移动端主界面布局 - 自动模式，无手动按钮"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 状态显示区域（顶部）
        self.status_label = Label(
            text='GuideVision\n正在启动...',
            font_size='24sp',
            halign='center',
            valign='middle',
            size_hint=(1, 0.12),
            pos_hint={'x': 0, 'top': 1},
            color=(1, 1, 1, 1),
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        self.add_widget(self.status_label)

        # 摄像头预览区域（主体）
        self.preview = Image(
            size_hint=(0.98, 0.7),
            pos_hint={'x': 0.01, 'top': 0.87},
            allow_stretch=True,
            keep_ratio=True,
        )
        self.add_widget(self.preview)

        # 底部信息栏
        self.info_label = Label(
            text='AI 自动识别场景中...\n语音交互：直接说话即可',
            font_size='18sp',
            halign='center',
            valign='middle',
            size_hint=(1, 0.15),
            pos_hint={'x': 0, 'y': 0},
            color=(0.8, 0.8, 0.8, 1),
        )
        self.info_label.bind(size=self.info_label.setter('text_size'))
        self.add_widget(self.info_label)


class GuideVisionMobileApp(App):
    """GuideVision 手机端应用 - 自动模式"""

    title = 'GuideVision 视觉辅助'
    icon = 'icon.png'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = None
        self._running = False
        self._frame_event = None
        self._camera_capture = None
        self._speech_recognizer = None

    def build(self):
        Window.clearcolor = (0.1, 0.1, 0.12, 1)
        if platform == "android":
            Window.softinput_mode = 'below_target'
        self.layout = GuideVisionMobileLayout()
        return self.layout

    def on_start(self):
        """应用启动后自动开始运行"""
        self.agent = AgentStateMachine()
        self.agent.register_callback(self._on_agent_message)
        self._init_voice()
        self._init_speech_recognition()
        self._speak("视觉辅助系统启动中，请稍候")
        # 延迟启动，等待权限生效
        Clock.schedule_once(self._auto_start, 2.0)

    def _auto_start(self, dt):
        """自动启动系统"""
        self._start()

    def _init_voice(self):
        """初始化语音"""
        self._plyer_tts = None
        if platform in ("android", "ios"):
            try:
                from plyer import tts
                self._plyer_tts = tts
                logger.info("Using plyer TTS")
            except Exception as e:
                logger.warning(f"plyer TTS not available: {e}")
        else:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 180)
                logger.info("Using pyttsx3 TTS")
            except Exception:
                self._engine = None

    def _speak(self, text):
        """语音播报"""
        if not text:
            return
        if self._plyer_tts:
            try:
                self._plyer_tts.speak(text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
        elif hasattr(self, '_engine') and self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass
        else:
            logger.info(f"[Voice] {text}")

    def _init_speech_recognition(self):
        """初始化语音识别"""
        self._stt = None
        self._speech_recognizer = None
        self._stt_poll_event = None

        if platform == "android":
            try:
                from plyer import stt
                self._stt = stt
                self._stt.language = "zh-CN"
                logger.info("Using plyer STT (Android)")
            except Exception as e:
                logger.warning(f"plyer STT not available: {e}")
        else:
            try:
                from utils.speech_recognition import SpeechRecognizer
                self._speech_recognizer = SpeechRecognizer()
                self._speech_recognizer.start_listening(self._on_voice_input)
                logger.info("Using SpeechRecognition (desktop)")
            except Exception as e:
                logger.warning(f"SpeechRecognition not available: {e}")

    def _start_listening(self):
        """开始语音监听"""
        if self._stt:
            # Android: 启动 STT 并轮询结果
            try:
                if not self._stt.listening:
                    self._stt.start()
                # 每 0.5 秒检查一次识别结果
                if self._stt_poll_event:
                    self._stt_poll_event.cancel()
                self._stt_poll_event = Clock.schedule_interval(self._poll_stt_results, 0.5)
                logger.info("STT listening started")
            except Exception as e:
                logger.error(f"STT start error: {e}")

    def _poll_stt_results(self, dt):
        """轮询 Android STT 识别结果"""
        if not self._stt or not self._running:
            return False
        try:
            results = self._stt.results
            if results:
                text = results[0] if isinstance(results, (list, tuple)) else str(results)
                text = text.strip()
                if text:
                    logger.info(f"STT result: {text}")
                    self._on_voice_input(text)
                    # 清除结果，继续监听
                    self._stt.results = []
            # 如果停止了监听，重新启动
            if not self._stt.listening:
                self._stt.start()
        except Exception as e:
            logger.debug(f"STT poll error: {e}")
        return True  # 继续轮询

    def _on_voice_input(self, text):
        """处理语音输入"""
        if not self.agent or not self._running:
            return
        logger.info(f"Voice input: {text}")
        response = self.agent.handle_voice_command(text)
        if response:
            self._speak(response)

    def _on_agent_message(self, message, priority):
        """Agent 回调：语音播报 + 更新界面"""
        if self._running:
            self._speak(message)
            self.layout.info_label.text = message

    def _init_camera(self):
        """初始化摄像头"""
        if HAS_CV2:
            try:
                self._camera_capture = cv2.VideoCapture(0)
                if self._camera_capture.isOpened():
                    self._camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    logger.info("OpenCV camera opened")
                    return True
            except Exception as e:
                logger.warning(f"OpenCV camera failed: {e}")

        if platform == "android":
            try:
                from android.camera import Camera
                logger.info("Using Android Camera API")
                return True
            except Exception as e:
                logger.warning(f"Android Camera API not available: {e}")

        logger.warning("No camera available, running in demo mode")
        return True

    def _start(self):
        if self._running:
            return

        if not self._init_camera():
            self.layout.status_label.text = "无法打开摄像头"
            return

        self.agent.start()
        self._running = True
        self._frame_event = Clock.schedule_interval(self._process_frame, 1.0 / 10.0)
        self._start_listening()
        self.layout.status_label.text = "AI 自动模式运行中"
        self.layout.info_label.text = "语音交互：直接说话即可\nAI 自动识别场景并切换模式"
        self._speak("系统已启动，AI 自动识别场景中")
        logger.info("Mobile app started")

    def _stop(self):
        self._running = False
        if self._frame_event:
            self._frame_event.cancel()
            self._frame_event = None
        if self._stt_poll_event:
            self._stt_poll_event.cancel()
            self._stt_poll_event = None
        if self._stt:
            try:
                self._stt.stop()
            except Exception:
                pass
        if self._camera_capture:
            self._camera_capture.release()
            self._camera_capture = None
        if self._speech_recognizer:
            self._speech_recognizer.stop_listening()
        if self.agent:
            self.agent.stop()
        self.layout.status_label.text = "系统已停止"
        self._speak("系统已停止")
        logger.info("Mobile app stopped")

    def _process_frame(self, dt):
        """处理视频帧"""
        if not self._running or not self.agent:
            return False

        frame = self._read_frame()
        if frame is None:
            return True

        try:
            message = self.agent.process_frame(frame)
            if message:
                self.layout.info_label.text = message
        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        # 更新预览
        if HAS_CV2:
            try:
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]),
                    colorfmt='bgr'
                )
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.layout.preview.texture = texture
            except Exception as e:
                logger.error(f"Preview error: {e}")

        return True

    def _read_frame(self):
        """读取一帧"""
        if self._camera_capture is None:
            return None

        if HAS_CV2 and self._camera_capture is not None:
            try:
                ret, frame = self._camera_capture.read()
                if ret and frame is not None:
                    return frame
            except Exception as e:
                logger.error(f"Camera read error: {e}")

        return None

    def on_pause(self):
        return True

    def on_resume(self):
        if self._running:
            self._start()

    def on_stop(self):
        self._stop()


def main():
    app = GuideVisionMobileApp()
    app.run()


if __name__ == "__main__":
    main()
