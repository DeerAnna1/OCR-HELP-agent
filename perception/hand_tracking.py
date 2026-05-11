"""
手部追踪模块 - 跨平台
Android 上 mediapipe 不可用时降级
"""
import logging
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

logger = logging.getLogger(__name__)


@dataclass
class HandInfo:
    detected: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    fingers_extended: int = 0
    is_reaching: bool = False
    is_grasping: bool = False
    confidence: float = 0.0


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


class HandTrackingModule:
    def __init__(self):
        self.landmarker = None
        self._init_mediapipe()

    def _get_model_path(self):
        import os
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if os.path.exists(model_path):
            return model_path
        os.makedirs(model_dir, exist_ok=True)
        logger.info("Downloading hand landmarker model...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path

    def _init_mediapipe(self):
        if not HAS_MEDIAPIPE:
            logger.warning("mediapipe not available. Hand tracking disabled.")
            return
        if not HAS_CV2:
            logger.warning("OpenCV not available. Hand tracking disabled.")
            return
        try:
            model_path = self._get_model_path()
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            logger.error(f"Failed to init MediaPipe Hands: {e}")

    def detect(self, frame):
        if self.landmarker is None or not HAS_CV2:
            return HandInfo()

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return HandInfo()

            hand = result.hand_landmarks[0]
            h, w = frame.shape[:2]

            xs = [int(lm.x * w) for lm in hand]
            ys = [int(lm.y * h) for lm in hand]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            fingers = self._count_fingers(hand)

            return HandInfo(
                detected=True,
                x=(x_min + x_max) // 2,
                y=(y_min + y_max) // 2,
                width=x_max - x_min,
                height=y_max - y_min,
                fingers_extended=fingers,
                is_reaching=fingers >= 3,
                is_grasping=fingers <= 1,
                confidence=0.8,
            )
        except Exception as e:
            logger.error(f"Hand tracking error: {e}")
            return HandInfo()

    def _count_fingers(self, landmarks):
        count = 0
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        if landmarks[tips[0]].x < landmarks[pips[0]].x:
            count += 1

        for i in range(1, 5):
            if landmarks[tips[i]].y < landmarks[pips[i]].y:
                count += 1

        return count
