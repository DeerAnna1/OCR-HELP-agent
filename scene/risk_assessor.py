import logging
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RISK_P0_DISTANCE_CM,
    RISK_P1_DISTANCE_CM,
    RISK_P2_DISTANCE_CM,
    RISK_P3_DISTANCE_CM,
)
from scene.spatial_analyzer import SceneGraph, SpatialObject

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    level: str
    priority: int
    message: str
    action: str
    should_interrupt: bool = False


class RiskAssessor:
    def assess(self, scene: SceneGraph) -> RiskAssessment:
        if scene.risk_level == "danger":
            return self._assess_p0(scene)
        if scene.risk_level == "warning":
            return self._assess_p1(scene)
        if scene.risk_level == "caution":
            return self._assess_p2(scene)
        return self._assess_p3(scene)

    def _assess_p0(self, scene: SceneGraph) -> RiskAssessment:
        obstacles = [
            o for o in scene.objects
            if o.risk == "critical" and o.distance_cm < RISK_P0_DISTANCE_CM
        ]
        if not obstacles:
            obstacles = [
                o for o in scene.objects
                if o.risk == "high" and o.distance_cm < RISK_P0_DISTANCE_CM
            ]
        if obstacles:
            nearest = min(obstacles, key=lambda o: o.distance_cm)
            return RiskAssessment(
                level="P0",
                priority=0,
                message=f"停下！{nearest.position}有{nearest.name_cn}！",
                action="stop",
                should_interrupt=True,
            )
        return RiskAssessment(level="P0", priority=0, message="停下！", action="stop", should_interrupt=True)

    def _assess_p1(self, scene: SceneGraph) -> RiskAssessment:
        obstacles = [
            o for o in scene.objects
            if o.risk in ("high", "critical") and o.distance_cm < RISK_P1_DISTANCE_CM
        ]
        if obstacles:
            nearest = min(obstacles, key=lambda o: o.distance_cm)
            return RiskAssessment(
                level="P1",
                priority=1,
                message=f"前方一步有{nearest.name_cn}",
                action="slow_down",
            )
        return RiskAssessment(level="P1", priority=1, message="前方有障碍，注意", action="slow_down")

    def _assess_p2(self, scene: SceneGraph) -> RiskAssessment:
        obstacles = [
            o for o in scene.objects
            if o.risk == "medium" and o.distance_cm < RISK_P2_DISTANCE_CM
        ]
        if not obstacles:
            return RiskAssessment(level="P2", priority=2, message="", action="continue")
        nearest = min(obstacles, key=lambda o: o.distance_cm)
        if scene.passable_direction:
            direction_text = "偏左走" if scene.passable_direction == "left" else "偏右走"
            return RiskAssessment(
                level="P2",
                priority=2,
                message=f"{nearest.position}有{nearest.name_cn}，{direction_text}",
                action="detour",
            )
        return RiskAssessment(
            level="P2",
            priority=2,
            message=f"{nearest.position}有{nearest.name_cn}，请小心",
            action="caution",
        )

    def _assess_p3(self, scene: SceneGraph) -> RiskAssessment:
        info_objects = [
            o for o in scene.objects
            if o.risk == "low" and o.distance_cm < RISK_P3_DISTANCE_CM
        ]
        if info_objects:
            nearest = min(info_objects, key=lambda o: o.distance_cm)
            return RiskAssessment(
                level="P3",
                priority=3,
                message=f"{nearest.position}是{nearest.name_cn}",
                action="inform",
            )
        return RiskAssessment(level="P3", priority=3, message="", action="continue")

    def should_announce(self, assessment: RiskAssessment) -> bool:
        return assessment.priority <= 2 and bool(assessment.message)

    def get_priority_text(self, assessment: RiskAssessment) -> str:
        priority_map = {
            0: "立即危险",
            1: "高风险",
            2: "中等风险",
            3: "信息提示",
        }
        return priority_map.get(assessment.priority, "未知")
