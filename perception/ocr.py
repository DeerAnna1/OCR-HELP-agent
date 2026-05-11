"""
OCR 模块 - 跨平台文字识别
Android 上使用降级模式
"""
import re
import logging
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class OCRResult:
    blocks: list = field(default_factory=list)
    full_text: str = ""
    summary: str = ""
    has_text: bool = False
    text_position: str = "unknown"

    @property
    def block_count(self) -> int:
        return len(self.blocks)


class OCRModule:
    MEDICINE_KEYWORDS = [
        "mg", "毫升", "每次", "每日", "有效期", "保质期", "生产日期",
        "用法", "用量", "一次", "两次", "三次", "口服", "外用",
        "片", "粒", "胶囊", "颗粒", "口服液",
    ]

    FOOD_KEYWORDS = [
        "配料", "营养成分", "能量", "蛋白质", "脂肪", "碳水",
        "钠", "保质期", "生产日期", "净含量",
    ]

    def __init__(self, languages=None):
        self.languages = languages or OCR_LANGUAGES
        self._lang_str = "+".join(self.languages)
        if not HAS_TESSERACT:
            logger.warning("pytesseract not available, OCR in fallback mode")
        if not HAS_CV2:
            logger.warning("OpenCV not available, OCR in fallback mode")

    def process(self, frame: np.ndarray) -> OCRResult:
        result = OCRResult()

        if HAS_CV2 and HAS_TESSERACT:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = self._preprocess(gray)
            result = self._tesseract_ocr(processed, frame)
        else:
            result = self._fallback_ocr(frame)

        if result.blocks:
            result.has_text = True
            result.text_position = self._estimate_position(result.blocks, frame.shape[1])
            result.summary = self._generate_summary(result.full_text)

        return result

    def _preprocess(self, gray):
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def _tesseract_ocr(self, processed, original):
        result = OCRResult()
        try:
            data = pytesseract.image_to_data(
                processed, lang=self._lang_str,
                output_type=pytesseract.Output.DICT
            )
            current_block = -1
            block_texts = []

            for i in range(len(data["text"])):
                block_num = data["block_num"][i]
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if conf < 30 or not text:
                    continue

                if block_num != current_block:
                    if block_texts:
                        combined = " ".join(block_texts)
                        result.full_text += combined + " "
                    current_block = block_num
                    block_texts = [text]
                    result.blocks.append(TextBlock(
                        text=text, x=data["left"][i], y=data["top"][i],
                        w=data["width"][i], h=data["height"][i], confidence=conf / 100.0,
                    ))
                else:
                    block_texts.append(text)

            if block_texts:
                result.full_text += " ".join(block_texts) + " "
            result.full_text = result.full_text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
        return result

    def _fallback_ocr(self, frame):
        logger.info("Using fallback OCR (no text extraction)")
        return OCRResult()

    def _estimate_position(self, blocks, frame_width):
        if not blocks:
            return "unknown"
        avg_x = sum(b.x + b.w / 2 for b in blocks) / len(blocks)
        ratio = avg_x / frame_width
        if ratio < 0.33:
            return "left"
        elif ratio > 0.67:
            return "right"
        return "center"

    def _generate_summary(self, text):
        if not text:
            return ""
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.MEDICINE_KEYWORDS):
            return self._summarize_medicine(text)
        if any(kw in text_lower for kw in self.FOOD_KEYWORDS):
            return self._summarize_food(text)
        return self._summarize_general(text)

    def _summarize_medicine(self, text):
        parts = []
        name_match = re.search(r"[\u4e00-\u9fff]{2,}(?:片|胶囊|颗粒|口服液|滴丸)", text)
        if name_match:
            parts.append(f"药品：{name_match.group()}")
        return "。".join(parts) if parts else text[:50]

    def _summarize_food(self, text):
        parts = []
        name_lines = text.split("\n")
        if name_lines:
            parts.append(f"产品：{name_lines[0][:20]}")
        return "。".join(parts) if parts else text[:50]

    def _summarize_general(self, text):
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) > 50:
            return clean[:50] + "..."
        return clean

    def check_alignment(self, frame):
        return "请将手机对准文字区域。"
