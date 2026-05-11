"""
物体检测模块 - 跨平台
Android 上 YOLO 不可用时使用轻量检测
"""
import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_MODEL, YOLO_CONFIDENCE, CAMERA_WIDTH

logger = logging.getLogger(__name__)

OBSTACLE_CLASSES = {
    "chair", "bench", "table", "door", "wall", "stairs",
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "dog", "cat", "backpack", "suitcase", "bottle", "cup",
}

GRABBABLE_CLASSES = {
    "cup", "bottle", "bowl", "phone", "remote", "book",
    "scissors", "knife", "fork", "spoon", "apple", "orange",
    "banana", "mouse", "keyboard", "pen", "pencil", "wallet",
    "keys", "medicine", "medicine_box", "chopsticks",
}

DANGEROUS_CLASSES = {"knife", "scissors", "car", "truck", "bus", "motorcycle", "bicycle"}

OBSTACLE_CN = {
    "chair": "椅子", "bench": "长椅", "table": "桌子", "door": "门",
    "wall": "墙壁", "person": "人", "stairs": "台阶", "car": "车",
    "bicycle": "自行车", "motorcycle": "摩托车", "dog": "狗", "cat": "猫",
    "backpack": "背包", "suitcase": "行李箱", "bottle": "瓶子",
    "cup": "杯子", "bowl": "碗",
}


@dataclass
class DetectedObject:
    class_name: str
    class_name_cn: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    position: str = "center"
    distance_hint: str = "medium"
    is_obstacle: bool = False
    is_grabbable: bool = False
    is_dangerous: bool = False

    def __post_init__(self):
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1


@dataclass
class DetectionResult:
    objects: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    grabbables: list = field(default_factory=list)
    dangerous: list = field(default_factory=list)
    frame_width: int = CAMERA_WIDTH
    frame_height: int = 480

    @property
    def has_obstacles(self) -> bool:
        return len(self.obstacles) > 0

    @property
    def has_grabbables(self) -> bool:
        return len(self.grabbables) > 0

    @property
    def has_dangerous(self) -> bool:
        return len(self.dangerous) > 0


class ObjectDetectionModule:
    def __init__(self, model_path=None, confidence=None):
        self.confidence = confidence or YOLO_CONFIDENCE
        self.model = None
        self._model_path = model_path or YOLO_MODEL
        self._load_model()

    def _load_model(self):
        if not HAS_YOLO:
            logger.warning("ultralytics not available. Detection disabled.")
            return
        try:
            self.model = YOLO(self._model_path)
            logger.info(f"Loaded YOLO model: {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(self, frame, target_class=None):
        h, w = frame.shape[:2]
        result = DetectionResult(frame_width=w, frame_height=h)

        if self.model is None:
            # Android 轻量检测：基于轮廓和颜色分析
            return self._detect_lightweight(frame, result)

        try:
            yolo_results = self.model(frame, conf=self.confidence, verbose=False)
            for yolo_result in yolo_results:
                boxes = yolo_result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if target_class and cls_name != target_class:
                        continue

                    obj = DetectedObject(
                        class_name=cls_name,
                        class_name_cn=OBSTACLE_CN.get(cls_name, cls_name),
                        confidence=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    )
                    obj.position = self._get_position(obj.center_x, w)
                    obj.distance_hint = self._estimate_distance(obj, h)
                    obj.is_obstacle = cls_name in OBSTACLE_CLASSES
                    obj.is_grabbable = cls_name in GRABBABLE_CLASSES
                    obj.is_dangerous = cls_name in DANGEROUS_CLASSES

                    result.objects.append(obj)
                    if obj.is_obstacle:
                        result.obstacles.append(obj)
                    if obj.is_grabbable:
                        result.grabbables.append(obj)
                    if obj.is_dangerous:
                        result.dangerous.append(obj)

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return result

    def _get_position(self, center_x, frame_width):
        ratio = center_x / frame_width
        if ratio < 0.33:
            return "left"
        elif ratio > 0.67:
            return "right"
        return "center"

    def _estimate_distance(self, obj, frame_height):
        obj_height_ratio = obj.height / frame_height
        bottom_ratio = obj.y2 / frame_height
        if bottom_ratio > 0.85 and obj_height_ratio > 0.3:
            return "very_near"
        if obj_height_ratio > 0.2:
            return "near"
        if obj_height_ratio > 0.08:
            return "medium"
        return "far"

    def find_target(self, frame, target_name):
        target_map = {
            "杯子": "cup", "水杯": "cup", "水瓶": "bottle", "瓶子": "bottle",
            "手机": "phone", "钥匙": "keys", "药盒": "medicine_box",
            "门把手": "door", "门": "door", "碗": "bowl",
        }
        english_name = target_map.get(target_name, target_name)
        result = self.detect(frame, target_class=english_name)
        if result.grabbables:
            return max(result.grabbables, key=lambda o: o.confidence)
        return None

    def _detect_lightweight(self, frame, result):
        """
        Android 轻量检测：基于 OpenCV 轮廓分析检测大块障碍物。
        不依赖 YOLO/PyTorch，仅用 numpy + 基础图像处理。
        """
        if not HAS_CV2:
            return result

        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 膨胀边缘以连接断裂区域
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = (h * w) * 0.01  # 最小面积：帧面积的 1%
            max_area = (h * w) * 0.6   # 最大面积：帧面积的 60%

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                x, y, cw, ch = cv2.boundingRect(contour)
                # 过滤太扁或太窄的区域
                aspect = cw / ch if ch > 0 else 0
                if aspect > 5 or aspect < 0.2:
                    continue

                # 根据位置和大小推断物体类型
                center_x = x + cw / 2
                center_y = y + ch / 2
                bottom_ratio = (y + ch) / h

                # 底部大块区域很可能是障碍物
                if bottom_ratio > 0.7 and cw > w * 0.15:
                    class_name = "obstacle"
                    class_name_cn = "障碍物"
                    is_obstacle = True
                elif cw < w * 0.3 and ch < h * 0.3:
                    class_name = "object"
                    class_name_cn = "物体"
                    is_obstacle = False
                else:
                    class_name = "wall"
                    class_name_cn = "墙壁"
                    is_obstacle = True

                obj = DetectedObject(
                    class_name=class_name,
                    class_name_cn=class_name_cn,
                    confidence=0.4,
                    x1=x, y1=y, x2=x + cw, y2=y + ch,
                )
                obj.position = self._get_position(center_x, w)
                obj.distance_hint = self._estimate_distance(obj, h)
                obj.is_obstacle = is_obstacle

                result.objects.append(obj)
                if obj.is_obstacle:
                    result.obstacles.append(obj)

        except Exception as e:
            logger.debug(f"Lightweight detection error: {e}")

        return result
