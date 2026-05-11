"""
深度估计模块 - 跨平台
Android 上使用启发式方法
"""
import logging
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class DepthInfo:
    estimated_cm: float = 0.0
    confidence: float = 0.0
    near: bool = False
    very_near: bool = False
    zone: str = "far"


class DepthEstimationModule:
    def __init__(self, use_midas=False):
        self.use_midas = use_midas and HAS_TORCH
        self.model = None
        if self.use_midas:
            self._load_midas()

    def _load_midas(self):
        if not HAS_TORCH:
            logger.warning("torch not available. MiDaS depth disabled.")
            self.use_midas = False
            return
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
        except Exception as e:
            logger.warning(f"Failed to load MiDaS: {e}. Using heuristic depth.")
            self.use_midas = False

    def estimate(self, frame, bbox=None):
        if self.use_midas and self.model is not None and HAS_CV2:
            return self._midas_depth(frame, bbox)
        return self._heuristic_depth(frame, bbox)

    def _midas_depth(self, frame, bbox):
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(rgb).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                region = depth_map[y1:y2, x1:x2]
                if region.size > 0:
                    median_depth = float(np.median(region))
                    return self._depth_to_info(median_depth, depth_map)

            center_depth = float(np.median(depth_map))
            return self._depth_to_info(center_depth, depth_map)
        except Exception as e:
            logger.error(f"MiDaS depth estimation error: {e}")
            return DepthInfo()

    def _heuristic_depth(self, frame, bbox):
        h, w = frame.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            vertical_pos = y2 / h
            obj_height = (y2 - y1) / h
        else:
            vertical_pos = 0.5
            obj_height = 0.1

        estimated_cm = self._estimate_cm(vertical_pos, obj_height)

        info = DepthInfo(
            estimated_cm=estimated_cm,
            confidence=0.4,
            near=estimated_cm < 100,
            very_near=estimated_cm < 50,
            zone=self._get_zone(estimated_cm),
        )
        return info

    def _estimate_cm(self, vertical_pos, obj_height):
        base_cm = 300 - (vertical_pos * 200)
        if obj_height > 0.3:
            base_cm *= 0.5
        elif obj_height > 0.15:
            base_cm *= 0.7
        return max(10, min(300, base_cm))

    def _depth_to_info(self, median_depth, depth_map):
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        depth_range = depth_max - depth_min if depth_max > depth_min else 1
        normalized = (median_depth - depth_min) / depth_range

        estimated_cm = 10 + (1 - normalized) * 290
        return DepthInfo(
            estimated_cm=estimated_cm,
            confidence=0.7,
            near=estimated_cm < 100,
            very_near=estimated_cm < 50,
            zone=self._get_zone(estimated_cm),
        )

    def _get_zone(self, cm):
        if cm < 50:
            return "very_near"
        elif cm < 100:
            return "near"
        elif cm < 200:
            return "medium"
        return "far"
