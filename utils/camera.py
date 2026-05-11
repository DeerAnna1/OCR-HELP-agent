import logging

import cv2
try:
    import numpy as np
except ImportError:
    np = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, camera_id: int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT, fps: int = CAMERA_FPS):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._is_opened = False

    def open(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._is_opened = True
            logger.info(f"Camera {self.camera_id} opened: {self.width}x{self.height}@{self.fps}fps")
            return True
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return True, frame
        return False, None

    def is_opened(self) -> bool:
        return self._is_opened

    def release(self):
        if self.cap:
            self.cap.release()
            self._is_opened = False
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
