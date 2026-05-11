"""
自动模式选择器 - 根据场景上下文自动切换模式
无需用户手动按键，AI 自主判断当前应该使用哪种模式
"""
import logging
import time

try:
    import numpy as np
except ImportError:
    np = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WorkMode

logger = logging.getLogger(__name__)


class AutoModeSelector:
    """根据场景自动选择最合适的工作模式"""

    def __init__(self):
        self.current_mode = WorkMode.WALK
        self._last_switch_time = 0
        self._min_switch_interval = 3.0  # 最短切换间隔（秒）
        self._voice_override_until = 0   # 语音命令覆盖时间
        self._voice_mode = None
        self._motion_history = []
        self._text_region_frames = 0
        self._stable_frames = 0  # 连续稳定帧数（用户静止）

    def on_voice_command(self, mode: WorkMode, target: str = ""):
        """语音命令覆盖：用户说话时立即切换模式"""
        self._voice_mode = mode
        self._voice_override_until = time.time() + 15.0  # 语音命令保持15秒
        self._last_switch_time = time.time()
        logger.info(f"Voice override: mode={mode.value}, target={target}")

    def select(self, scene, detection, hand_info=None) -> WorkMode:
        """
        根据当前场景选择模式。

        优先级：
        1. 语音命令覆盖（15秒内有效）
        2. 障碍物检测 → WALK
        3. 文字区域检测 → READ
        4. 默认 → WALK（持续避障）
        """
        now = time.time()

        # 1. 语音命令覆盖优先
        if self._voice_mode and now < self._voice_override_until:
            self.current_mode = self._voice_mode
            return self.current_mode
        elif self._voice_mode and now >= self._voice_override_until:
            self._voice_mode = None  # 过期清除

        # 切换间隔保护：避免频繁切换
        if now - self._last_switch_time < self._min_switch_interval:
            return self.current_mode

        # 2. 检测到危险障碍物 → WALK（最高优先级）
        if scene.risk_level in ("danger", "warning"):
            if self.current_mode != WorkMode.WALK:
                self._switch(WorkMode.WALK, now)
            return self.current_mode

        # 3. 检测到前方有障碍物 → WALK
        if scene.has_obstacle_ahead and scene.risk_level == "caution":
            if self.current_mode != WorkMode.WALK:
                self._switch(WorkMode.WALK, now)
            return self.current_mode

        # 4. 检测到文字且用户静止 → READ
        has_text = self._detect_text_region(detection)
        if has_text and self._is_user_stable():
            if self.current_mode != WorkMode.READ:
                self._text_region_frames += 1
                if self._text_region_frames >= 15:  # 连续15帧检测到文字
                    self._switch(WorkMode.READ, now)
                    self._text_region_frames = 0
            return self.current_mode
        else:
            self._text_region_frames = 0

        # 5. 用户静止且无障碍物 → ASK（可以回答问题）
        if self._is_user_stable() and scene.risk_level == "safe":
            if len(detection.objects) > 0 and self.current_mode == WorkMode.WALK:
                # 有检测到物体但没有危险，可以切到 ASK 描述场景
                pass  # 保持 WALK，等用户主动说话

        # 6. 默认：WALK（持续避障）
        if self.current_mode not in (WorkMode.WALK,):
            self._switch(WorkMode.WALK, now)

        return self.current_mode

    def _switch(self, mode: WorkMode, now: float):
        """切换模式并记录"""
        old = self.current_mode
        self.current_mode = mode
        self._last_switch_time = now
        if old != mode:
            logger.info(f"Auto mode switch: {old.value} -> {mode.value}")

    def _detect_text_region(self, detection) -> bool:
        """
        简单判断帧中是否有文字区域。
        检测方法：如果检测到的对象中有细长形状（可能是文字标签），返回 True。
        """
        for obj in detection.objects:
            aspect = obj.width / obj.height if obj.height > 0 else 1
            # 文字通常宽高比 > 2 或 < 0.5
            if aspect > 2.5 or (aspect < 0.4 and obj.height > 30):
                return True
            # 小型密集物体可能是文字
            if obj.width < 100 and obj.height < 50 and obj.confidence > 0.3:
                return True
        return False

    def _is_user_stable(self) -> bool:
        """用户是否保持静止（基于运动历史）"""
        return self._stable_frames >= 30  # 约1秒（30fps）

    def update_motion(self, frame_diff: float):
        """
        更新运动状态。frame_diff 是当前帧与上一帧的差异度（0-1）。
        由 process_frame() 在每帧调用。
        """
        self._motion_history.append(frame_diff)
        if len(self._motion_history) > 60:
            self._motion_history.pop(0)

        # 如果最近几帧变化很小，认为用户静止
        recent = self._motion_history[-10:] if len(self._motion_history) >= 10 else self._motion_history
        avg_motion = sum(recent) / len(recent) if recent else 1.0

        if avg_motion < 0.02:  # 阈值：非常小的变化
            self._stable_frames += 1
        else:
            self._stable_frames = 0
