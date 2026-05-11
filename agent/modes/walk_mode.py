import logging

from scene.spatial_analyzer import SceneGraph
from scene.risk_assessor import RiskAssessor, RiskAssessment

logger = logging.getLogger(__name__)


class WalkMode:
    def __init__(self):
        self.last_risk_level = "safe"
        self.silence_frames = 0

    def process(self, scene: SceneGraph, risk: RiskAssessment) -> str | None:
        if risk.level == "P0":
            self.last_risk_level = "P0"
            self.silence_frames = 0
            return risk.message

        if risk.level == "P1":
            if self.last_risk_level != "P1":
                self.last_risk_level = "P1"
                self.silence_frames = 0
                return risk.message
            self.silence_frames += 1
            if self.silence_frames > 30:
                self.silence_frames = 0
                return risk.message
            return None

        if risk.level == "P2":
            if self.last_risk_level in ("P0", "P1"):
                self.last_risk_level = "P2"
                self.silence_frames = 0
                return risk.message
            if self.last_risk_level != "P2":
                self.last_risk_level = "P2"
                self.silence_frames = 0
                return risk.message
            self.silence_frames += 1
            if self.silence_frames > 60:
                self.silence_frames = 0
                return risk.message
            return None

        if self.last_risk_level != "safe":
            self.last_risk_level = "safe"
            self.silence_frames = 0
            return "可以继续"

        self.silence_frames += 1
        return None

    def reset(self):
        self.last_risk_level = "safe"
        self.silence_frames = 0
