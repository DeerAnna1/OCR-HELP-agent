import logging

try:
    import numpy as np
except ImportError:
    np = None

from perception.object_detection import DetectionResult
from perception.depth_estimation import DepthEstimationModule
from scene.spatial_analyzer import SceneGraph, SpatialAnalyzer

logger = logging.getLogger(__name__)


class FindMode:
    def __init__(self):
        self.found = False
        self.last_message = ""
        self.scan_direction = 1

    def process(
        self,
        frame: np.ndarray,
        target: str,
        detection: DetectionResult,
        scene: SceneGraph,
        depth_estimator: DepthEstimationModule,
    ) -> str | None:
        target_map = {
            "杯子": "cup", "水杯": "cup", "水瓶": "bottle", "瓶子": "bottle",
            "手机": "phone", "钥匙": "keys", "药盒": "medicine_box",
            "门把手": "door", "门": "door", "碗": "bowl",
        }
        english_target = target_map.get(target, target)

        matches = [
            obj for obj in detection.objects
            if obj.class_name == english_target or obj.class_name_cn == target
        ]

        if not matches:
            self.found = False
            if self.last_message != "scanning":
                self.last_message = "scanning"
                return f"正在扫描，未发现{target}，请缓慢移动手机"
            return None

        best = max(matches, key=lambda o: o.confidence)
        self.found = True

        position_text = self._get_position_text(best.position)
        depth_info = depth_estimator.estimate(frame, (best.x1, best.y1, best.x2, best.y2))
        distance_text = self._format_distance(depth_info.estimated_cm)

        message = f"发现{target}，在{position_text}，{distance_text}"

        if message != self.last_message:
            self.last_message = message
            return message
        return None

    def _get_position_text(self, position: str) -> str:
        mapping = {"left": "左前方", "center": "正前方", "right": "右前方"}
        return mapping.get(position, "正前方")

    def _format_distance(self, distance_cm: float) -> str:
        if distance_cm < 30:
            return "很近，手可及"
        if distance_cm < 50:
            return f"约{int(distance_cm)}厘米"
        if distance_cm < 100:
            return "约半臂距离"
        if distance_cm < 200:
            return "约一步远"
        return "较远"
