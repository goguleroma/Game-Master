"""
HUD parsing with PaddleOCR (primary) and Tesseract (fallback).

Extracts text, numbers, and structured data from game HUD regions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class OCREngine(Enum):
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    AUTO = "auto"


@dataclass
class OCRConfig:
    """Configuration for OCR engine."""
    engine: OCREngine = OCREngine.AUTO
    languages: List[str] = field(default_factory=lambda: ["en"])
    use_gpu: bool = True
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    cls_model_dir: Optional[str] = None
    # PaddleOCR specific
    use_angle_cls: bool = True
    drop_score: float = 0.5
    # Tesseract specific
    tesseract_config: str = "--oem 3 --psm 6"
    # Preprocessing
    preprocess: bool = True
    upscale_factor: float = 2.0
    # Regions to ignore (x, y, w, h)
    ignore_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    engine: str
    raw_result: Any = None


@dataclass
class HUDData:
    """Structured HUD data extracted from OCR."""
    # Common game HUD elements
    health: Optional[float] = None
    max_health: Optional[float] = None
    mana: Optional[float] = None
    max_mana: Optional[float] = None
    stamina: Optional[float] = None
    level: Optional[int] = None
    experience: Optional[int] = None
    gold: Optional[int] = None
    score: Optional[int] = None
    time_remaining: Optional[str] = None
    ammo: Optional[int] = None
    max_ammo: Optional[int] = None
    kills: Optional[int] = None
    deaths: Optional[int] = None
    # Custom key-value pairs
    custom: Dict[str, Any] = field(default_factory=dict)
    # Raw OCR results
    raw_results: List[OCRResult] = field(default_factory=list)


class OCREngineBase:
    """Base class for OCR engines."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    def initialize(self) -> bool:
        raise NotImplementedError
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        raise NotImplementedError
    
    def shutdown(self) -> None:
        pass


class PaddleOCREngine(OCREngineBase):
    """PaddleOCR engine wrapper."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self._ocr: Optional[PaddleOCR] = None
    
    def initialize(self) -> bool:
        if not PADDLEOCR_AVAILABLE:
            logger.error("PaddleOCR not installed")
            return False
        
        try:
            self._ocr = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.languages[0] if self.config.languages else "en",
                use_gpu=self.config.use_gpu,
                det_model_dir=self.config.det_model_dir,
                rec_model_dir=self.config.rec_model_dir,
                cls_model_dir=self.config.cls_model_dir,
                show_log=False,
            )
            logger.info("PaddleOCR initialized successfully")
            return True
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            return False
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if self._ocr is None:
            return []
        
        try:
            # Preprocess if needed
            if self.config.preprocess:
                image = self._preprocess(image)
            
            result = self._ocr.ocr(image, cls=self.config.use_angle_cls)
            
            ocr_results = []
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, confidence) = line
                    # Convert bbox to x, y, w, h
                    pts = np.array(bbox, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts) if CV2_AVAILABLE else self._bbox_to_rect(bbox)
                    
                    if confidence >= self.config.drop_score:
                        ocr_results.append(OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            engine="paddleocr",
                            raw_result=line,
                        ))
            
            return ocr_results
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            return []
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR."""
        if not CV2_AVAILABLE:
            return image
        
        # Upscale
        if self.config.upscale_factor != 1.0:
            h, w = image.shape[:2]
            new_w, new_h = int(w * self.config.upscale_factor), int(h * self.config.upscale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to 3-channel for PaddleOCR
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _bbox_to_rect(self, bbox: List[List[int]]) -> Tuple[int, int, int, int]:
        """Convert 4-point bbox to x, y, w, h."""
        pts = np.array(bbox, dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def shutdown(self) -> None:
        self._ocr = None


class TesseractEngine(OCREngineBase):
    """Tesseract OCR engine wrapper."""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self._initialized = False
    
    def initialize(self) -> bool:
        if not TESSERACT_AVAILABLE:
            logger.error("Tesseract not installed")
            return False
        
        try:
            # Test tesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            return False
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if not self._initialized:
            return []
        
        try:
            if self.config.preprocess:
                image = self._preprocess(image)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                config=self.config.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
            
            ocr_results = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                if not text:
                    continue
                
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=(x, y, w, h),
                    engine="tesseract",
                    raw_result=data,
                ))
            
            return ocr_results
        except Exception as e:
            logger.error(f"Tesseract recognition failed: {e}")
            return []
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for Tesseract."""
        if not CV2_AVAILABLE:
            return image
        
        # Upscale
        if self.config.upscale_factor != 1.0:
            h, w = image.shape[:2]
            new_w, new_h = int(w * self.config.upscale_factor), int(h * self.config.upscale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def shutdown(self) -> None:
        self._initialized = False


class HUDParser:
    """
    Parse OCR results into structured HUD data.
    
    Uses regex patterns to extract common game HUD elements.
    """
    
    # Common patterns for HUD elements
    PATTERNS = {
        "health": [
            r"(?:HP|Health|health)\s*[:\/]?\s*(\d+)\s*[\/\|]\s*(\d+)",
            r"(\d+)\s*\/\s*(\d+)\s*(?:HP|Health)",
        ],
        "mana": [
            r"(?:MP|Mana|mana)\s*[:\/]?\s*(\d+)\s*[\/\|]\s*(\d+)",
            r"(\d+)\s*\/\s*(\d+)\s*(?:MP|Mana)",
        ],
        "stamina": [
            r"(?:SP|Stamina|stamina)\s*[:\/]?\s*(\d+)\s*[\/\|]\s*(\d+)",
        ],
        "level": [
            r"(?:LV|Lv|Level|level)\s*[:\.]?\s*(\d+)",
        ],
        "experience": [
            r"(?:EXP|Exp|XP|experience)\s*[:\/]?\s*(\d+)\s*[\/\|]\s*(\d+)",
        ],
        "gold": [
            r"(?:Gold|gold|G)\s*[:\:]?\s*([\d,]+)",
            r"([\d,]+)\s*(?:Gold|gold|G)\b",
        ],
        "score": [
            r"(?:Score|score)\s*[:\:]?\s*([\d,]+)",
        ],
        "ammo": [
            r"(?:Ammo|ammo)\s*[:\/]?\s*(\d+)\s*[\/\|]\s*(\d+)",
            r"(\d+)\s*\/\s*(\d+)\s*(?:Ammo|ammo)",
        ],
        "kills": [
            r"(?:Kills|kills|K)\s*[:\:]?\s*(\d+)",
        ],
        "deaths": [
            r"(?:Deaths|deaths|D)\s*[:\:]?\s*(\d+)",
        ],
        "time": [
            r"(?:Time|time)\s*[:\:]?\s*(\d+:\d+)",
            r"(\d+:\d+)\s*(?:Time|time|Remaining)",
        ],
    }
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        self.patterns = self.PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Compile patterns
        self._compiled = {}
        for key, pat_list in self.patterns.items():
            self._compiled[key] = [re.compile(p, re.IGNORECASE) for p in pat_list]
    
    def parse(self, ocr_results: List[OCRResult]) -> HUDData:
        """Parse OCR results into structured HUD data."""
        # Combine all text
        full_text = " ".join(r.text for r in ocr_results)
        
        hud = HUDData(raw_results=ocr_results)
        
        # Map pattern names to HUDData attribute names
        field_mapping = {
            "time": "time_remaining",
        }
        
        # Extract using patterns
        for field_name, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(full_text)
                if match:
                    groups = match.groups()
                    attr_name = field_mapping.get(field_name, field_name)
                    if len(groups) >= 2:
                        # Two groups: current/max
                        try:
                            current = float(groups[0].replace(",", ""))
                            max_val = float(groups[1].replace(",", ""))
                            setattr(hud, attr_name, current)
                            setattr(hud, f"max_{attr_name}", max_val)
                        except ValueError:
                            pass
                    elif len(groups) == 1:
                        # Special handling for time fields - keep as string
                        if field_name == "time":
                            setattr(hud, attr_name, groups[0])
                        else:
                            try:
                                val = float(groups[0].replace(",", ""))
                                if field_name in ["level", "gold", "score", "kills", "deaths", "ammo"]:
                                    val = int(val)
                                setattr(hud, attr_name, val)
                            except ValueError:
                                pass
                    break
        
        # Try to extract any key: value pairs not caught by patterns
        self._extract_key_value(full_text, hud)
        
        return hud
    
    def _extract_key_value(self, text: str, hud: HUDData) -> None:
        """Extract generic key: value pairs."""
        # Pattern for "Key: Value" or "Key = Value"
        kv_pattern = re.compile(r"(\w+)\s*[:\=]\s*([\w\d\.\,\:\/]+)", re.IGNORECASE)
        for match in kv_pattern.finditer(text):
            key = match.group(1).lower()
            value = match.group(2)
            
            # Skip already parsed fields
            if hasattr(hud, key) and getattr(hud, key) is not None:
                continue
            
            # Try to convert value
            try:
                if "/" in value:
                    parts = value.split("/")
                    hud.custom[key] = {"current": float(parts[0]), "max": float(parts[1])}
                elif ":" in value and value.count(":") == 1:
                    hud.custom[key] = value  # Time format
                elif "," in value:
                    hud.custom[key] = int(value.replace(",", ""))
                elif "." in value:
                    hud.custom[key] = float(value)
                else:
                    hud.custom[key] = int(value)
            except ValueError:
                hud.custom[key] = value


class OCRProcessor:
    """
    Main OCR processor combining engine and parser.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        self.config = config or OCRConfig()
        self._engine: Optional[OCREngineBase] = None
        self._parser = HUDParser()
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the selected OCR engine."""
        if self._initialized:
            return True
        
        engine_type = self.config.engine
        
        # Auto-select engine
        if engine_type == OCREngine.AUTO:
            if PADDLEOCR_AVAILABLE:
                engine_type = OCREngine.PADDLEOCR
            elif TESSERACT_AVAILABLE:
                engine_type = OCREngine.TESSERACT
            else:
                logger.error("No OCR engine available")
                return False
        
        # Create engine
        if engine_type == OCREngine.PADDLEOCR:
            self._engine = PaddleOCREngine(self.config)
        elif engine_type == OCREngine.TESSERACT:
            self._engine = TesseractEngine(self.config)
        else:
            logger.error(f"Unknown engine: {engine_type}")
            return False
        
        success = self._engine.initialize()
        if success:
            self._initialized = True
            logger.info(f"OCR initialized with {engine_type.value}")
        return success
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Run OCR on image."""
        if not self._initialized:
            if not self.initialize():
                return []
        
        return self._engine.recognize(image)
    
    def extract_hud(self, image: np.ndarray) -> HUDData:
        """Extract structured HUD data from image."""
        results = self.recognize(image)
        return self._parser.parse(results)
    
    def extract_text_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
    ) -> Dict[Tuple[int, int, int, int], str]:
        """Extract text from specific regions."""
        results = {}
        for region in regions:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                results[region] = ""
                continue
            
            ocr_results = self.recognize(roi)
            text = " ".join(r.text for r in ocr_results)
            results[region] = text
        
        return results
    
    def shutdown(self) -> None:
        if self._engine:
            self._engine.shutdown()
        self._initialized = False


def create_ocr_processor(
    engine: OCREngine = OCREngine.AUTO,
    languages: List[str] = None,
    use_gpu: bool = True,
) -> OCRProcessor:
    """Factory function to create OCR processor."""
    config = OCRConfig(
        engine=engine,
        languages=languages or ["en"],
        use_gpu=use_gpu,
    )
    return OCRProcessor(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test OCR processor creation
    ocr = create_ocr_processor()
    print(f"PaddleOCR available: {PADDLEOCR_AVAILABLE}")
    print(f"Tesseract available: {TESSERACT_AVAILABLE}")
    print(f"OpenCV available: {CV2_AVAILABLE}")
    
    if ocr.initialize():
        print("OCR initialized successfully")
        
        # Test with dummy image
        dummy = np.ones((100, 300, 3), dtype=np.uint8) * 255
        # Add some text-like pattern
        if CV2_AVAILABLE:
            cv2.putText(dummy, "HP: 100/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(dummy, "Gold: 1,234", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        hud = ocr.extract_hud(dummy)
        print(f"Extracted HUD: health={hud.health}, max_health={hud.max_health}, gold={hud.gold}")
        print(f"Custom: {hud.custom}")
    else:
        print("OCR initialization failed")