"""
Real state extractor implementation for MiniGrid and generic games.

Combines screen capture, preprocessing, OCR, and object detection
to extract structured game state from observations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from core.types import GameState, Observation

logger = logging.getLogger(__name__)

# Import perception modules
try:
    from perception.capture import ScreenCapture, CaptureConfig, FrameRingBuffer
    from perception.preprocessing import PreprocessingPipeline, PreprocessConfig, DeviceType
    from perception.ocr import OCRProcessor, OCRConfig, OCREngine, HUDData, HUDParser
    from perception.detection import (
        DetectionConfig, Detection, DetectionResult,
        ModelType, create_detector, draw_detections
    )
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Perception modules not fully available: {e}")
    PERCEPTION_AVAILABLE = False


@dataclass
class StateExtractorConfig:
    """Configuration for state extraction."""
    # Capture
    capture_config: Optional[CaptureConfig] = None
    # Preprocessing
    preprocess_config: Optional[PreprocessConfig] = None
    # OCR
    ocr_config: Optional[OCRConfig] = None
    # Detection
    detection_config: Optional[DetectionConfig] = None
    # HUD regions (x, y, w, h) relative to capture
    hud_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    # Attention region for focused processing
    attention_region: Optional[Tuple[int, int, int, int]] = None
    # Enable/disable components
    use_ocr: bool = True
    use_detection: bool = True
    use_preprocessing: bool = True
    # Frame differencing
    use_frame_diff: bool = True
    diff_threshold: float = 10.0
    # MiniGrid specific
    minigrid_mode: bool = False
    minigrid_cell_size: int = 32


class StateExtractor:
    """
    Real state extractor combining multiple perception modalities.
    
    Extracts structured GameState from raw observations using:
    - Screen capture (dxcam/mss)
    - GPU preprocessing (torch-directml)
    - OCR for HUD parsing (PaddleOCR/Tesseract)
    - Object detection (YOLO/RT-DETR ONNX)
    - Frame differencing for change detection
    """
    
    def __init__(self, config: Optional[StateExtractorConfig] = None) -> None:
        self.config = config or StateExtractorConfig()
        
        # Components
        self._capture: Optional[ScreenCapture] = None
        self._preprocessor: Optional[PreprocessingPipeline] = None
        self._ocr: Optional[OCRProcessor] = None
        self._detector = None
        self._hud_parser = HUDParser()
        
        # State
        self._initialized = False
        self._frame_count = 0
        self._last_hud: Optional[HUDData] = None
        self._last_detections: List[Detection] = []
        
        # Performance tracking
        self._timings: Dict[str, List[float]] = {
            "capture": [],
            "preprocess": [],
            "ocr": [],
            "detection": [],
            "total": [],
        }
        
        logger.info("StateExtractor created")
    
    def initialize(self) -> bool:
        """Initialize all perception components."""
        if self._initialized:
            return True
        
        if not PERCEPTION_AVAILABLE:
            logger.error("Perception modules not available")
            return False
        
        success = True
        
        # Initialize capture
        if self.config.capture_config:
            self._capture = ScreenCapture(self.config.capture_config)
            if not self._capture.start():
                logger.error("Failed to start capture")
                success = False
        
        # Initialize preprocessor
        if self.config.use_preprocessing and self.config.preprocess_config:
            self._preprocessor = PreprocessingPipeline(self.config.preprocess_config)
            logger.info(f"Preprocessor on {self._preprocessor.device.value}")
        
        # Initialize OCR
        if self.config.use_ocr and self.config.ocr_config:
            self._ocr = OCRProcessor(self.config.ocr_config)
            if not self._ocr.initialize():
                logger.warning("OCR initialization failed")
                self._ocr = None
        
        # Initialize detector
        if self.config.use_detection and self.config.detection_config:
            self._detector = create_detector(self.config.detection_config)
            if not self._detector.initialize():
                logger.warning("Detector initialization failed")
                self._detector = None
        
        # Set attention region on capture
        if self._capture and self.config.attention_region:
            x, y, w, h = self.config.attention_region
            self._capture.set_attention_region(x, y, w, h)
        
        self._initialized = success
        if success:
            logger.info("StateExtractor initialized successfully")
        else:
            logger.error("StateExtractor initialization failed")
        
        return success
    
    def extract(self, observation: Observation) -> GameState:
        """
        Extract GameState from observation.
        
        This is the main entry point implementing ObservationAdapter protocol.
        """
        total_start = time.perf_counter()
        
        # Get image from observation
        image = self._get_image(observation)
        
        if image is None:
            return GameState(
                objects=["no_image"],
                hud={"status": "no_image"},
                goal="wait_for_frame",
                confidence=0.0,
            )
        
        # Capture frame if using live capture
        if self._capture and self.config.capture_config:
            captured = self._capture.get_frame()
            if captured is not None:
                image = captured
        
        # Frame differencing
        changed_regions = []
        if self.config.use_frame_diff and self._capture:
            changed_regions = self._capture.get_changed_regions(threshold=self.config.diff_threshold)
        
        # Preprocess
        processed_image = image
        if self._preprocessor and self.config.use_preprocessing:
            preprocess_start = time.perf_counter()
            processed_image = self._preprocessor.process(image)
            self._timings["preprocess"].append((time.perf_counter() - preprocess_start) * 1000)
        
        # Extract objects via detection
        objects = []
        detections = []
        detection_confidence = 0.0
        
        if self._detector and self.config.use_detection:
            det_start = time.perf_counter()
            det_result: DetectionResult = self._detector.detect(image, self._frame_count)
            detections = det_result.detections
            detection_confidence = np.mean([d.confidence for d in detections]) if detections else 0.0
            objects = [f"{d.class_name}:{d.confidence:.2f}" for d in detections]
            self._last_detections = detections
            self._timings["detection"].append(det_result.inference_time_ms)
        
        # Extract HUD via OCR
        hud_data = {}
        ocr_confidence = 0.0
        
        if self._ocr and self.config.use_ocr:
            ocr_start = time.perf_counter()
            hud: HUDData = self._ocr.extract_hud(image)
            self._last_hud = hud
            
            # Convert HUDData to dict
            hud_data = {
                "health": hud.health,
                "max_health": hud.max_health,
                "mana": hud.mana,
                "max_mana": hud.max_mana,
                "stamina": hud.stamina,
                "level": hud.level,
                "experience": hud.experience,
                "gold": hud.gold,
                "score": hud.score,
                "time": hud.time_remaining,
                "ammo": hud.ammo,
                "max_ammo": hud.max_ammo,
                "kills": hud.kills,
                "deaths": hud.deaths,
                "custom": hud.custom,
            }
            # Remove None values
            hud_data = {k: v for k, v in hud_data.items() if v is not None}
            
            ocr_confidence = np.mean([r.confidence for r in hud.raw_results]) if hud.raw_results else 0.0
            self._timings["ocr"].append((time.perf_counter() - ocr_start) * 1000)
        
        # MiniGrid specific extraction
        if self.config.minigrid_mode:
            minigrid_state = self._extract_minigrid(image)
            objects.extend(minigrid_state.get("objects", []))
            hud_data.update(minigrid_state.get("hud", {}))
        
        # Determine goal from state
        goal = self._infer_goal(objects, hud_data, detections)
        
        # Overall confidence
        confidence = max(detection_confidence, ocr_confidence, 0.3 if objects else 0.1)
        
        # Add timing info to HUD
        hud_data["_timings"] = {k: np.mean(v[-10:]) if v else 0 for k, v in self._timings.items()}
        hud_data["_fps"] = self._capture.current_fps if self._capture else 0
        hud_data["_changed_regions"] = len(changed_regions)
        
        self._frame_count += 1
        self._timings["total"].append((time.perf_counter() - total_start) * 1000)
        
        return GameState(
            objects=objects,
            hud=hud_data,
            goal=goal,
            confidence=confidence,
            controls=[],  # Could be learned
        )
    
    def _get_image(self, observation: Observation) -> Optional[np.ndarray]:
        """Extract image from observation."""
        image = observation.image
        
        if image is None:
            return None
        
        if isinstance(image, np.ndarray):
            return image
        
        if isinstance(image, dict):
            # Try to get frame from dict
            if "frame" in image:
                return image["frame"]
            if "image" in image:
                return image["image"]
        
        return None
    
    def _extract_minigrid(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract MiniGrid-specific state from image."""
        # MiniGrid has a grid of cells, each potentially containing an object
        # This is a simplified version - real implementation would use template matching
        # or a trained classifier for MiniGrid entities
        
        cell_size = self.config.minigrid_cell_size
        h, w = image.shape[:2]
        
        grid_w = w // cell_size
        grid_h = h // cell_size
        
        objects = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                cell = image[gy*cell_size:(gy+1)*cell_size, gx*cell_size:(gx+1)*cell_size]
                # Simple color-based detection (placeholder)
                mean_color = cell.mean(axis=(0, 1))
                if mean_color[0] > 200:  # Red-ish
                    objects.append(f"red_object_at_{gx}_{gy}")
                elif mean_color[1] > 200:  # Green-ish
                    objects.append(f"green_object_at_{gx}_{gy}")
                elif mean_color[2] > 200:  # Blue-ish
                    objects.append(f"blue_object_at_{gx}_{gy}")
        
        return {"objects": objects, "hud": {"grid_size": f"{grid_w}x{grid_h}"}}
    
    def _infer_goal(self, objects: List[str], hud: Dict, detections: List[Detection]) -> str:
        """Infer current goal from state."""
        # Simple heuristic goal inference
        if hud.get("health", 100) < 30:
            return "survive"
        if hud.get("gold", 0) > 0:
            return "collect"
        if any("enemy" in obj.lower() or "monster" in obj.lower() for obj in objects):
            return "combat"
        if any("door" in obj.lower() or "exit" in obj.lower() for obj in objects):
            return "escape"
        if detections:
            return "explore"
        return "observe"
    
    def get_capture(self) -> Optional[ScreenCapture]:
        """Get capture instance for direct access."""
        return self._capture
    
    def get_preprocessor(self) -> Optional[PreprocessingPipeline]:
        """Get preprocessor instance."""
        return self._preprocessor
    
    def get_ocr(self) -> Optional[OCRProcessor]:
        """Get OCR processor."""
        return self._ocr
    
    def get_detector(self):
        """Get detector instance."""
        return self._detector
    
    def set_attention_region(self, x: int, y: int, w: int, h: int) -> None:
        """Update attention region."""
        self.config.attention_region = (x, y, w, h)
        if self._capture:
            self._capture.set_attention_region(x, y, w, h)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from capture (implements ObservationAdapter protocol)."""
        if self._capture:
            return self._capture.get_frame()
        return None
    
    def get_frame_with_metadata(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get latest frame with timestamp and frame_id."""
        if self._capture:
            return self._capture.get_frame_with_metadata()
        return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            "avg_capture_ms": np.mean(self._timings["capture"][-100:]) if self._timings["capture"] else 0,
            "avg_preprocess_ms": np.mean(self._timings["preprocess"][-100:]) if self._timings["preprocess"] else 0,
            "avg_ocr_ms": np.mean(self._timings["ocr"][-100:]) if self._timings["ocr"] else 0,
            "avg_detection_ms": np.mean(self._timings["detection"][-100:]) if self._timings["detection"] else 0,
            "avg_total_ms": np.mean(self._timings["total"][-100:]) if self._timings["total"] else 0,
            "fps": self._capture.current_fps if self._capture else 0,
            "frames_processed": self._frame_count,
        }
    
    def shutdown(self) -> None:
        """Shutdown all components."""
        if self._capture:
            self._capture.stop()
        if self._ocr:
            self._ocr.shutdown()
        if self._detector:
            self._detector.shutdown()
        self._initialized = False
        logger.info("StateExtractor shutdown")
    
    def __enter__(self) -> "StateExtractor":
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


class SimpleStateExtractor:
    """
    Simple state extractor for testing and simulation.
    
    Handles dict observations with game state keys directly.
    """
    
    def extract(self, observation: Observation) -> GameState:
        image = observation.image
        
        if image is None:
            return GameState(
                objects=["unknown"],
                hud={"status": "no_data"},
                goal=None,
                confidence=0.0,
            )
        
        if isinstance(image, dict):
            objects = image.get("objects", [])
            hud = image.get("hud", {})
            goal = image.get("goal")
            confidence = image.get("confidence", 0.0)
            return GameState(
                objects=objects,
                hud=hud,
                goal=goal,
                confidence=confidence,
                controls=image.get("controls", []),
            )
        
        return GameState(
            objects=["visible_scene"],
            hud={"status": "unknown"},
            goal="observe",
            confidence=0.3,
        )


def create_state_extractor(
    capture_fps: int = 30,
    capture_resolution: Tuple[int, int] = (1920, 1080),
    use_directml: bool = True,
    detection_model: Optional[str] = None,
    minigrid_mode: bool = False,
) -> StateExtractor:
    """Factory function to create a configured StateExtractor."""
    
    config = StateExtractorConfig(
        capture_config=CaptureConfig(
            target_fps=capture_fps,
            width=capture_resolution[0],
            height=capture_resolution[1],
            use_gpu_preprocess=use_directml,
        ),
        preprocess_config=PreprocessConfig(
            target_size=(640, 640),
            device=DeviceType.DIRECTML if use_directml else DeviceType.CPU,
            half_precision=use_directml,
        ) if use_directml else None,
        ocr_config=OCRConfig(
            engine=OCREngine.AUTO,
            use_gpu=use_directml,
        ),
        detection_config=DetectionConfig(
            model_path=detection_model or "models/yolov8n.onnx",
            model_type=ModelType.YOLOV8,
            use_directml=use_directml,
            confidence_threshold=0.3,
        ) if detection_model else None,
        minigrid_mode=minigrid_mode,
    )
    
    return StateExtractor(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test creation
    extractor = create_state_extractor(
        capture_fps=30,
        capture_resolution=(1920, 1080),
        use_directml=True,
        minigrid_mode=False,
    )
    
    print("StateExtractor created")
    print(f"Perception available: {PERCEPTION_AVAILABLE}")