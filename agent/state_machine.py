import time
import logging
from dataclasses import dataclass, field

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WorkMode, DEFAULT_ANNOUNCE_INTERVAL
from perception import OCRModule, ObjectDetectionModule, HandTrackingModule, DepthEstimationModule
from scene import SpatialAnalyzer, RiskAssessor
from agent.modes import FindMode, GrabMode, WalkMode, ReadMode, AskMode
from agent.auto_mode import AutoModeSelector

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    current_mode: WorkMode = WorkMode.WALK
    is_active: bool = False
    target_object: str = ""
    last_announce_time: float = 0.0
    announce_interval: float = DEFAULT_ANNOUNCE_INTERVAL
    last_message: str = ""
    grab_state: str = "searching"
    frame_count: int = 0


class AgentStateMachine:
    def __init__(self):
        self.state = AgentState()
        self.ocr = OCRModule()
        self.detector = ObjectDetectionModule()
        self.hand_tracker = HandTrackingModule()
        self.depth_estimator = DepthEstimationModule()
        self.spatial_analyzer = SpatialAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.find_mode = FindMode()
        self.grab_mode = GrabMode()
        self.walk_mode = WalkMode()
        self.read_mode = ReadMode()
        self.ask_mode = AskMode()
        self.auto_mode = AutoModeSelector()
        self._callbacks: list = []
        self._prev_frame_gray = None

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def _emit(self, message: str, priority: int = 3):
        now = time.time()
        if priority <= 1 or (now - self.state.last_announce_time) >= self.state.announce_interval:
            self.state.last_announce_time = now
            self.state.last_message = message
            for cb in self._callbacks:
                cb(message, priority)

    def set_mode(self, mode: WorkMode, target: str = ""):
        self.state.current_mode = mode
        self.state.target_object = target
        self.state.grab_state = "searching"
        mode_names = {
            WorkMode.FIND: "找物",
            WorkMode.GRAB: "抓取",
            WorkMode.WALK: "行走",
            WorkMode.READ: "读文字",
            WorkMode.ASK: "问答",
        }
        name = mode_names.get(mode, "未知")
        if target:
            self._emit(f"已进入{name}模式，目标：{target}", priority=2)
        else:
            self._emit(f"已进入{name}模式", priority=2)

    def start(self):
        self.state.is_active = True
        self._emit("系统已启动")

    def stop(self):
        self.state.is_active = False
        self._emit("系统已停止")

    def process_frame(self, frame: np.ndarray) -> str | None:
        if not self.state.is_active:
            return None

        self.state.frame_count += 1
        h, w = frame.shape[:2]

        # 计算帧差异用于运动检测
        gray = np.mean(frame, axis=2).astype(np.float32)
        if self._prev_frame_gray is not None:
            diff = np.mean(np.abs(gray - self._prev_frame_gray)) / 255.0
            self.auto_mode.update_motion(diff)
        self._prev_frame_gray = gray.copy()

        detection_result = self.detector.detect(frame, target_class=None)
        hand_info = self.hand_tracker.detect(frame)

        scene = self.spatial_analyzer.analyze(detection_result.objects)
        risk = self.risk_assessor.assess(scene)

        # 自动选择模式（P0 风险仍优先）
        if not (risk.should_interrupt and risk.priority == 0):
            self.auto_mode.select(scene, detection_result, hand_info)
            if self.state.current_mode != self.auto_mode.current_mode:
                self.state.current_mode = self.auto_mode.current_mode

        if risk.should_interrupt and risk.priority == 0:
            self._emit(risk.message, priority=0)
            return risk.message

        mode = self.state.current_mode
        message = None

        if mode == WorkMode.FIND:
            message = self.find_mode.process(
                frame, self.state.target_object, detection_result,
                scene, self.depth_estimator
            )
        elif mode == WorkMode.GRAB:
            message = self.grab_mode.process(
                frame, self.state.target_object, detection_result,
                hand_info, scene, self.depth_estimator
            )
            if self.grab_mode.is_complete:
                self.state.grab_state = "complete"
        elif mode == WorkMode.WALK:
            message = self.walk_mode.process(scene, risk)
        elif mode == WorkMode.READ:
            ocr_result = self.ocr.process(frame)
            message = self.read_mode.process(ocr_result)
        elif mode == WorkMode.ASK:
            ocr_result = self.ocr.process(frame)
            message = self.ask_mode.process(
                detection_result, ocr_result, scene, self.state.target_object
            )

        if message:
            self._emit(message, priority=risk.priority)

        return message

    def handle_voice_command(self, command: str) -> str | None:
        command = command.strip().lower()

        if any(kw in command for kw in ["找", "帮我找", "在哪", "在哪里"]):
            target = self._extract_target(command)
            if target:
                self.set_mode(WorkMode.FIND, target)
                self.auto_mode.on_voice_command(WorkMode.FIND, target)
                return f"开始寻找{target}"

        if any(kw in command for kw in ["抓", "拿", "取", "帮我拿"]):
            target = self._extract_target(command)
            if target:
                self.set_mode(WorkMode.GRAB, target)
                self.auto_mode.on_voice_command(WorkMode.GRAB, target)
                return f"开始抓取{target}"

        if any(kw in command for kw in ["读", "读一下", "看看", "什么字"]):
            self.set_mode(WorkMode.READ)
            self.auto_mode.on_voice_command(WorkMode.READ)
            return "开始读文字"

        if any(kw in command for kw in ["走路", "行走", "避障", "导航"]):
            self.set_mode(WorkMode.WALK)
            self.auto_mode.on_voice_command(WorkMode.WALK)
            return "已进入行走模式"

        if any(kw in command for kw in ["这是什么", "前面是什么", "什么在前面"]):
            self.set_mode(WorkMode.ASK, command)
            self.auto_mode.on_voice_command(WorkMode.ASK, command)
            return "正在查看"

        if any(kw in command for kw in ["停", "停止", "结束"]):
            self.stop()
            return "已停止"

        if any(kw in command for kw in ["开始", "启动"]):
            self.start()
            return "已启动"

        # 默认：当作问题处理
        self.set_mode(WorkMode.ASK, command)
        self.auto_mode.on_voice_command(WorkMode.ASK, command)
        return "正在理解你的问题"

    def _extract_target(self, command: str) -> str:
        targets = [
            "水杯", "杯子", "水瓶", "瓶子", "手机", "钥匙",
            "药盒", "门把手", "门", "碗", "遥控器", "眼镜",
        ]
        for t in targets:
            if t in command:
                return t
        words = command.replace("帮我找", "").replace("帮我拿", "").replace("在哪", "").strip()
        return words if words else ""
