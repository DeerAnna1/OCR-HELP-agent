from enum import Enum

try:
    from kivy.utils import platform as _kivy_platform
    _IS_MOBILE = _kivy_platform in ("android", "ios")
except ImportError:
    _IS_MOBILE = False


class WorkMode(str, Enum):
    FIND = "find"
    GRAB = "grab"
    WALK = "walk"
    READ = "read"
    ASK = "ask"


class RiskLevel(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

OCR_LANGUAGES = ["chi_sim", "eng"]
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5

DEPTH_NEAR_CM = 30
DEPTH_MID_CM = 100
DEPTH_FAR_CM = 300

RISK_P0_DISTANCE_CM = 50
RISK_P1_DISTANCE_CM = 100
RISK_P2_DISTANCE_CM = 200
RISK_P3_DISTANCE_CM = 300

DIRECTION_LEFT_THRESHOLD = 0.33
DIRECTION_RIGHT_THRESHOLD = 0.67

VIBRATION_ENABLED = True
VIBRATION_SERIAL_PORT = "/dev/ttyUSB0"
VIBRATION_BAUD_RATE = 9600

TTS_RATE = 180
TTS_VOLUME = 0.9

MIN_ANNOUNCE_INTERVAL = 0.5
DEFAULT_ANNOUNCE_INTERVAL = 1.0

GRAB_GUIDE_THRESHOLD_CM = 5
