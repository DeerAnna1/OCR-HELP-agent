import logging
from dataclasses import dataclass, field

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DIRECTION_LEFT_THRESHOLD,
    DIRECTION_RIGHT_THRESHOLD,
    DEPTH_NEAR_CM,
    DEPTH_MID_CM,
    DEPTH_FAR_CM,
)
from perception.object_detection import DetectedObject

logger = logging.getLogger(__name__)


@dataclass
class SpatialObject:
    name: str
    name_cn: str
    position: str
    distance_cm: float
    distance_desc: str
    orientation: str = "unknown"
    reachable: bool = False
    risk: str = "low"
    confidence: float = 0.0

    def to_speech(self) -> str:
        parts = [f"{self.name_cn}在{self.position}"]
        parts.append(f"约{self._format_distance()}")
        if self.orientation != "unknown":
            parts.append(self.orientation)
        return "，".join(parts)

    def _format_distance(self) -> str:
        if self.distance_cm < 30:
            return "很近，手可及"
        if self.distance_cm < 50:
            return f"{int(self.distance_cm)}厘米"
        if self.distance_cm < 100:
            return "半臂距离"
        if self.distance_cm < 200:
            return "一步左右"
        return "较远"


@dataclass
class SceneGraph:
    objects: list[SpatialObject] = field(default_factory=list)
    has_obstacle_ahead: bool = False
    passable_direction: str | None = None
    risk_level: str = "safe"

    def get_object(self, name: str) -> SpatialObject | None:
        for obj in self.objects:
            if name in obj.name or name in obj.name_cn:
                return obj
        return None

    def get_obstacles(self) -> list[SpatialObject]:
        return [o for o in self.objects if o.risk in ("medium", "high", "critical")]

    def get_nearest_obstacle(self) -> SpatialObject | None:
        obstacles = self.get_obstacles()
        if not obstacles:
            return None
        return min(obstacles, key=lambda o: o.distance_cm)


class SpatialAnalyzer:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def analyze(
        self,
        detected_objects: list[DetectedObject],
        depth_map: np.ndarray | None = None,
    ) -> SceneGraph:
        scene = SceneGraph()

        for obj in detected_objects:
            spatial_obj = self._to_spatial(obj, depth_map)
            scene.objects.append(spatial_obj)

        obstacles = scene.get_obstacles()
        scene.has_obstacle_ahead = any(
            o.position == "center" and o.distance_cm < DEPTH_MID_CM
            for o in obstacles
        )

        if scene.has_obstacle_ahead:
            left_clear = not any(
                o.position == "left" and o.distance_cm < DEPTH_MID_CM
                for o in obstacles
            )
            right_clear = not any(
                o.position == "right" and o.distance_cm < DEPTH_MID_CM
                for o in obstacles
            )
            if left_clear:
                scene.passable_direction = "left"
            elif right_clear:
                scene.passable_direction = "right"

        scene.risk_level = self._assess_risk_level(scene)
        return scene

    def _to_spatial(self, obj: DetectedObject, depth_map: np.ndarray | None) -> SpatialObject:
        distance_cm = self._estimate_distance(obj, depth_map)
        position = self._get_position_text(obj.position)
        distance_desc = self._get_distance_desc(distance_cm)
        orientation = self._estimate_orientation(obj)
        risk = self._assess_object_risk(obj, distance_cm)
        reachable = distance_cm < 50

        return SpatialObject(
            name=obj.class_name,
            name_cn=obj.class_name_cn,
            position=position,
            distance_cm=distance_cm,
            distance_desc=distance_desc,
            orientation=orientation,
            reachable=reachable,
            risk=risk,
            confidence=obj.confidence,
        )

    def _estimate_distance(self, obj: DetectedObject, depth_map: np.ndarray | None) -> float:
        if depth_map is not None:
            cx = int(obj.center_x)
            cy = int(obj.center_y)
            h, w = depth_map.shape[:2]
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            region_size = 5
            x1 = max(0, cx - region_size)
            x2 = min(w, cx + region_size)
            y1 = max(0, cy - region_size)
            y2 = min(h, cy + region_size)
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                return float(np.median(region))

        obj_height_ratio = obj.height / self.frame_height
        bottom_ratio = obj.y2 / self.frame_height

        if bottom_ratio > 0.8 and obj_height_ratio > 0.3:
            return 30.0
        if obj_height_ratio > 0.2:
            return 50.0
        if obj_height_ratio > 0.1:
            return 100.0
        if obj_height_ratio > 0.05:
            return 200.0
        return 300.0

    def _get_position_text(self, position: str) -> str:
        mapping = {"left": "左前方", "center": "正前方", "right": "右前方"}
        return mapping.get(position, "正前方")

    def _get_distance_desc(self, distance_cm: float) -> str:
        if distance_cm < 30:
            return "很近"
        if distance_cm < 50:
            return "半步内"
        if distance_cm < 100:
            return "一臂内"
        if distance_cm < 200:
            return "一步左右"
        return "三步外"

    def _estimate_orientation(self, obj: DetectedObject) -> str:
        aspect = obj.width / obj.height if obj.height > 0 else 1
        if obj.class_name in ("cup", "bottle", "bowl"):
            return "杯口朝上" if aspect < 1.5 else "可能倾斜"
        return "unknown"

    def _assess_object_risk(self, obj: DetectedObject, distance_cm: float) -> str:
        if obj.is_dangerous and distance_cm < DEPTH_NEAR_CM:
            return "critical"
        if obj.is_dangerous:
            return "high"
        if obj.is_obstacle and distance_cm < DEPTH_NEAR_CM:
            return "high"
        if obj.is_obstacle and distance_cm < DEPTH_MID_CM:
            return "medium"
        return "low"

    def _assess_risk_level(self, scene: SceneGraph) -> str:
        if any(o.risk == "critical" for o in scene.objects):
            return "danger"
        if any(o.risk == "high" for o in scene.objects):
            return "warning"
        if any(o.risk == "medium" for o in scene.objects):
            return "caution"
        return "safe"
