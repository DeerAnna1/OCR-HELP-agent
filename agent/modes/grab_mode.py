import logging

try:
    import numpy as np
except ImportError:
    np = None

from perception.object_detection import DetectionResult
from perception.hand_tracking import HandInfo, HandTrackingModule
from perception.depth_estimation import DepthEstimationModule
from scene.spatial_analyzer import SceneGraph

logger = logging.getLogger(__name__)


class GrabMode:
    GRAB_STATES = ("searching", "found", "guiding", "approaching", "ready", "complete")

    def __init__(self):
        self.state = "searching"
        self.target_obj = None
        self.guide_count = 0
        self.is_complete = False
        self.hand_tracker = HandTrackingModule()

    def process(
        self,
        frame: np.ndarray,
        target: str,
        detection: DetectionResult,
        hand: HandInfo,
        scene: SceneGraph,
        depth_estimator: DepthEstimationModule,
    ) -> str | None:
        if self.state == "searching":
            return self._search(target, detection, depth_estimator, frame)
        if self.state == "found":
            return self._found(target)
        if self.state == "guiding":
            return self._guide(hand, detection, target, depth_estimator, frame)
        if self.state == "approaching":
            return self._approach(hand, detection, target)
        if self.state == "ready":
            return self._ready(hand)
        return None

    def _search(self, target: str, detection: DetectionResult, depth_estimator: DepthEstimationModule, frame: np.ndarray) -> str | None:
        target_map = {
            "杯子": "cup", "水杯": "cup", "水瓶": "bottle", "瓶子": "bottle",
            "手机": "phone", "钥匙": "keys", "药盒": "medicine_box",
            "门把手": "door", "碗": "bowl",
        }
        english_target = target_map.get(target, target)
        matches = [
            obj for obj in detection.objects
            if obj.class_name == english_target or obj.class_name_cn == target
        ]
        if not matches:
            return f"正在寻找{target}，请保持手机稳定"
        best = max(matches, key=lambda o: o.confidence)
        self.target_obj = best
        self.state = "found"
        position = self._get_position_text(best.position)
        depth_info = depth_estimator.estimate(frame, (best.x1, best.y1, best.x2, best.y2))
        distance_text = self._format_distance(depth_info.estimated_cm)
        return f"找到{target}，在{position}，{distance_text}"

    def _found(self, target: str) -> str | None:
        self.state = "guiding"
        self.guide_count = 0
        return f"请伸手，我会引导你靠近{target}"

    def _guide(self, hand: HandInfo, detection: DetectionResult, target: str, depth_estimator: DepthEstimationModule, frame: np.ndarray) -> str | None:
        if self.target_obj is None:
            self.state = "searching"
            return f"目标丢失，重新寻找{target}"

        if not hand.detected:
            self.guide_count += 1
            if self.guide_count > 3:
                return "没有检测到手，请将手伸入画面"
            return None

        target_cx = int(self.target_obj.center_x)
        target_cy = int(self.target_obj.center_y)
        guidance = self.hand_tracker.get_guidance(hand, target_cx, target_cy)

        depth_info = depth_estimator.estimate(frame, (self.target_obj.x1, self.target_obj.y1, self.target_obj.x2, self.target_obj.y2))
        if depth_info.estimated_cm < 30:
            self.state = "approaching"
            return f"手已经很接近{target}，慢一点"

        self.guide_count += 1
        if guidance and guidance != "继续":
            return guidance
        if self.guide_count % 5 == 0:
            return f"继续向{target}靠近"
        return None

    def _approach(self, hand: HandInfo, detection: DetectionResult, target: str) -> str | None:
        if not hand.detected:
            return None

        if self.target_obj is None:
            self.state = "searching"
            return f"目标丢失，重新寻找{target}"

        target_cx = int(self.target_obj.center_x)
        target_cy = int(self.target_obj.center_y)
        guidance = self.hand_tracker.get_guidance(hand, target_cx, target_cy)

        if guidance and "接近" in guidance:
            self.state = "ready"
            return f"已经接近{target}，可以小心抓取"

        self.guide_count += 1
        if self.guide_count % 3 == 0:
            return "慢一点，继续靠近"
        return None

    def _ready(self, hand: HandInfo) -> str | None:
        if hand.detected and hand.is_grasping:
            self.state = "complete"
            self.is_complete = True
            return "抓取成功！请注意杯中可能有水，保持水平"
        if not hand.detected:
            return None
        return "已经很近了，慢慢合拢手指"

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
        return "较远"

    def reset(self):
        self.state = "searching"
        self.target_obj = None
        self.guide_count = 0
        self.is_complete = False
