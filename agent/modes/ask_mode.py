import logging

from perception.object_detection import DetectionResult
from perception.ocr import OCRResult
from scene.spatial_analyzer import SceneGraph

logger = logging.getLogger(__name__)


class AskMode:
    def __init__(self):
        self.last_answer = ""

    def process(
        self,
        detection: DetectionResult,
        ocr_result: OCRResult,
        scene: SceneGraph,
        question: str = "",
    ) -> str | None:
        if not question:
            return None

        answer = self._answer(question, detection, ocr_result, scene)
        if answer and answer != self.last_answer:
            self.last_answer = answer
            return answer
        return None

    def _answer(
        self,
        question: str,
        detection: DetectionResult,
        ocr_result: OCRResult,
        scene: SceneGraph,
    ) -> str | None:
        q = question.lower()

        if any(kw in q for kw in ["前面能不能走", "可以走吗", "能走吗", "路通吗"]):
            if scene.risk_level == "danger":
                return "不能走，前方有危险，请停下"
            if scene.risk_level == "warning":
                return "前方有障碍，请小心"
            if scene.passable_direction:
                direction = "左边" if scene.passable_direction == "left" else "右边"
                return f"前方有障碍，{direction}可以通过"
            return "前方看起来可以通行，但请使用盲杖确认"

        if any(kw in q for kw in ["这是什么", "什么东西", "啥东西"]):
            if detection.objects:
                names = [o.class_name_cn for o in detection.objects[:3]]
                return f"我看到了{', '.join(names)}"
            return "没有识别到明确的物体，请靠近一点"

        if any(kw in q for kw in ["有没有空座位", "座位", "椅子"]):
            chairs = [o for o in detection.objects if o.class_name == "chair"]
            if chairs:
                chair = chairs[0]
                position = self._get_position_text(chair.position)
                return f"看到一把椅子在{position}"
            return "没有看到空椅子"

        if any(kw in q for kw in ["哪个是", "哪一个是"]):
            target = q.replace("哪个是", "").replace("哪一个是", "").strip()
            matches = [o for o in detection.objects if target in o.class_name_cn]
            if matches:
                obj = matches[0]
                position = self._get_position_text(obj.position)
                return f"{target}在{position}"
            return f"没有找到{target}"

        if any(kw in q for kw in ["满了吗", "有没有水", "装满了吗"]):
            cups = [o for o in detection.objects if o.class_name in ("cup", "bowl", "bottle")]
            if cups:
                return "检测到容器，但无法判断是否装满"
            return "没有检测到容器"

        if any(kw in q for kw in ["写的什么", "什么字", "读一下"]):
            if ocr_result.has_text and ocr_result.summary:
                return ocr_result.summary
            return "没有检测到文字"

        if detection.objects:
            obj = detection.objects[0]
            position = self._get_position_text(obj.position)
            return f"前方{position}有{obj.class_name_cn}"

        return "我不太确定，请再描述一下"

    def _get_position_text(self, position: str) -> str:
        mapping = {"left": "左前方", "center": "正前方", "right": "右前方"}
        return mapping.get(position, "正前方")

    def reset(self):
        self.last_answer = ""
