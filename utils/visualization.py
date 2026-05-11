import cv2
try:
    import numpy as np
except ImportError:
    np = None
from PIL import Image, ImageDraw, ImageFont

from perception.object_detection import DetectionResult
from perception.hand_tracking import HandInfo
from scene.spatial_analyzer import SceneGraph


RISK_COLORS = {
    "dangerous": (0, 0, 255),
    "obstacle": (0, 165, 255),
    "grabbable": (0, 255, 255),
    "safe": (0, 255, 0),
}

# Map WorkMode enum values to display strings
MODE_LABELS = {
    "FIND": "寻找",
    "GRAB": "抓取",
    "WALK": "行走",
    "READ": "阅读",
    "ASK":  "问答",
}

RISK_CN = {
    "safe": "安全",
    "caution": "注意",
    "warning": "警告",
    "danger": "危险",
}

_font = None


def _get_font(size=20):
    global _font
    if _font is None:
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        for path in candidates:
            try:
                _font = ImageFont.truetype(path, size)
                break
            except (OSError, IOError):
                continue
        if _font is None:
            _font = ImageFont.load_default()
    return _font


def _put_text_cn(frame, text, pos, color=(255, 255, 255), size=20):
    """Draw text (including Chinese) on a cv2 frame using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text(pos, text, font=font, fill=color)
    if np is not None:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def _get_risk_key(obj) -> str:
    if obj.is_dangerous:
        return "dangerous"
    if obj.is_obstacle:
        return "obstacle"
    if obj.is_grabbable:
        return "grabbable"
    return "safe"


def draw_detections(frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
    vis = frame.copy()
    for obj in detection.objects:
        color = RISK_COLORS.get(_get_risk_key(obj), (0, 255, 0))
        cv2.rectangle(vis, (obj.x1, obj.y1), (obj.x2, obj.y2), color, 2)
        label = f"{obj.class_name_cn} {obj.confidence:.2f}"
        vis = _put_text_cn(vis, label, (obj.x1, obj.y1 - 22), color, size=18)
    return vis


def draw_hand(frame: np.ndarray, hand: HandInfo) -> np.ndarray:
    vis = frame.copy()
    if hand.detected:
        x1 = hand.x - hand.width // 2
        y1 = hand.y - hand.height // 2
        x2 = hand.x + hand.width // 2
        y2 = hand.y + hand.height // 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"手 ({hand.fingers_extended}指)"
        vis = _put_text_cn(vis, label, (x1, y1 - 22), (255, 0, 0), size=18)
    return vis


def draw_scene_info(
    frame: np.ndarray,
    scene: SceneGraph,
    message: str = "",
    mode_name: str = "",
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    risk_color = {
        "safe": (0, 255, 0),
        "caution": (0, 255, 255),
        "warning": (0, 165, 255),
        "danger": (0, 0, 255),
    }.get(scene.risk_level, (255, 255, 255))

    # Bottom info bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    if message:
        vis = _put_text_cn(vis, message[:40], (10, h - 55), (255, 255, 255), size=18)

    risk_cn = RISK_CN.get(scene.risk_level, scene.risk_level)
    risk_text = f"风险: {risk_cn}"
    vis = _put_text_cn(vis, risk_text, (10, h - 28), risk_color, size=18)

    obj_count = len(scene.objects)
    obj_text = f"物体: {obj_count}"
    vis = _put_text_cn(vis, obj_text, (w - 120, h - 28), (255, 255, 255), size=18)

    # Top-left mode indicator
    if mode_name:
        label = MODE_LABELS.get(mode_name, mode_name)
        cv2.rectangle(vis, (5, 5), (200, 35), (0, 0, 0), -1)
        vis = _put_text_cn(vis, label, (10, 10), (0, 255, 255), size=22)

    return vis


def draw_full_debug(
    frame: np.ndarray,
    detection: DetectionResult,
    hand: HandInfo,
    scene: SceneGraph,
    message: str = "",
    mode_name: str = "",
) -> np.ndarray:
    vis = draw_detections(frame, detection)
    vis = draw_hand(vis, hand)
    vis = draw_scene_info(vis, scene, message, mode_name)
    return vis
