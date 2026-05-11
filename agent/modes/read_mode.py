import logging

from perception.ocr import OCRResult

logger = logging.getLogger(__name__)


class ReadMode:
    def __init__(self):
        self.last_summary = ""
        self.read_full = False
        self.no_text_frames = 0

    def process(self, ocr_result: OCRResult) -> str | None:
        if not ocr_result.has_text:
            self.no_text_frames += 1
            if self.no_text_frames == 10:
                return "没有检测到文字，请将手机对准文字区域"
            if self.no_text_frames > 30:
                self.no_text_frames = 11
            return None

        self.no_text_frames = 0

        alignment_hint = None
        if ocr_result.text_position == "left":
            alignment_hint = "文字在画面左侧，请手机向左一点"
        elif ocr_result.text_position == "right":
            alignment_hint = "文字在画面右侧，请手机向右一点"

        if alignment_hint and self.last_summary != alignment_hint:
            self.last_summary = alignment_hint
            return alignment_hint

        summary = ocr_result.summary
        if summary and summary != self.last_summary:
            self.last_summary = summary
            self.read_full = False
            return f"检测到文字。{summary}"

        if self.read_full and ocr_result.full_text:
            self.read_full = False
            return ocr_result.full_text[:200]

        return None

    def request_full_text(self) -> bool:
        self.read_full = True
        return True

    def reset(self):
        self.last_summary = ""
        self.read_full = False
        self.no_text_frames = 0
