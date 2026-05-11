"""
GuideVision - 视觉辅助 Agent
桌面端入口 / Android 入口分发
"""
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("GuideVision")

# ── Android 检测：直接启动 Kivy 移动端 UI ──────────────
def _is_android():
    try:
        from jnius import autoclass
        return True
    except ImportError:
        pass
    try:
        from kivy.utils import platform
        return platform == "android"
    except ImportError:
        return False

if _is_android():
    # Android: 启动 Kivy 移动端
    from main_mobile import main
    main()
    sys.exit(0)


# ── 桌面端逻辑 ────────────────────────────────────────
import time
import argparse

from platform_adapter import CameraAdapter, VoiceAdapter, VibrationAdapter, SpeechRecognizerAdapter, IS_MOBILE
from config import WorkMode
from agent.state_machine import AgentStateMachine


class GuideVisionApp:
    def __init__(self, camera_id: int = 0, no_gui: bool = False, no_voice: bool = False):
        self.camera = CameraAdapter(camera_id)
        self.agent = AgentStateMachine()
        self.voice = VoiceAdapter()
        self.vibration = VibrationAdapter()
        self.speech_recognizer = SpeechRecognizerAdapter()
        self.no_gui = no_gui
        self.no_voice = no_voice
        self._running = False
        self._last_message = ""
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.agent.register_callback(self._on_agent_message)

    def _on_agent_message(self, message: str, priority: int):
        if message == self._last_message:
            return
        self._last_message = message

        if not self.no_voice:
            self.voice.speak(message, priority)

        if priority == 0:
            self.vibration.vibrate_direction("stop")
        elif priority == 1:
            self.vibration.vibrate_direction("danger")
        elif "左" in message:
            self.vibration.vibrate_direction("left")
        elif "右" in message:
            self.vibration.vibrate_direction("right")

    def start(self):
        if not self.camera.open():
            logger.error("无法打开摄像头，退出")
            return

        self.agent.start()

        if not self.no_voice:
            self.voice.speak("视觉辅助系统已启动", priority=2)

        self.speech_recognizer.start_listening(self._on_voice_input)

        self._running = True
        logger.info("GuideVision started.")

        if IS_MOBILE:
            logger.info("Running in mobile mode")
            return

        try:
            self._main_loop_desktop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.agent.stop()
        self.voice.stop()
        self.vibration.stop()
        self.speech_recognizer.stop_listening()
        self.camera.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
        logger.info("GuideVision stopped")

    def _main_loop_desktop(self):
        import cv2
        while self._running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue

            message = self.agent.process_frame(frame)

            if not self.no_gui:
                self._draw_gui(frame, message)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def _draw_gui(self, frame, message):
        import cv2
        from utils.visualization import draw_full_debug
        detection = self.agent.detector.detect(frame)
        hand = self.agent.hand_tracker.detect(frame)
        scene = self.agent.spatial_analyzer.analyze(detection.objects)
        mode_name = self.agent.state.current_mode.name if hasattr(self.agent.state.current_mode, 'name') else str(self.agent.state.current_mode)
        vis = draw_full_debug(frame, detection, hand, scene, message or self._last_message, mode_name)
        cv2.imshow("GuideVision", vis)

    def process_frame(self, frame):
        """供移动端调用的帧处理方法"""
        if not self._running:
            return None
        return self.agent.process_frame(frame)

    def _on_voice_input(self, text: str):
        logger.info(f"Voice input: {text}")
        response = self.agent.handle_voice_command(text)
        if response and not self.no_voice:
            self.voice.speak(response, priority=2)


def main():
    parser = argparse.ArgumentParser(description="GuideVision - 视觉辅助 Agent")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--no-gui", action="store_true", help="不显示 GUI 窗口")
    parser.add_argument("--no-voice", action="store_true", help="不播放语音")

    args = parser.parse_args()

    app = GuideVisionApp(
        camera_id=args.camera,
        no_gui=args.no_gui,
        no_voice=args.no_voice,
    )

    app.start()


if __name__ == "__main__":
    main()
