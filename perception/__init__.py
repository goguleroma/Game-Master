"""
Perception package for Game Master.

Provides real-time screen capture, preprocessing, OCR, and object detection
for game state extraction at 30fps @ 1080p on AMD RX 6800 XT.
"""

from perception.capture import (
    ScreenCapture,
    CaptureConfig,
    FrameMetrics,
    FrameRingBuffer,
    capture_once,
)

from perception.state_extractor import (
    StateExtractor,
    StateExtractorConfig,
    create_state_extractor,
)

from perception.preprocessing import (
    PreprocessingPipeline,
    PreprocessConfig,
    DeviceType,
    create_detection_pipeline,
    create_classification_pipeline,
    create_ocr_pipeline,
    compute_frame_diff,
    get_motion_regions,
)

from perception.ocr import (
    OCRProcessor,
    OCRConfig,
    OCREngine,
    HUDData,
    HUDParser,
    create_ocr_processor,
)

from perception.detection import (
    DetectionConfig,
    Detection,
    DetectionResult,
    ModelType,
    create_detector,
    draw_detections,
    BaseDetector,
    YOLODetector,
    RTDETRDetector,
)

__all__ = [
    # Capture
    "ScreenCapture",
    "CaptureConfig",
    "FrameMetrics",
    "FrameRingBuffer",
    "capture_once",
    # State extraction
    "StateExtractor",
    "StateExtractorConfig",
    "create_state_extractor",
    # Preprocessing
    "PreprocessingPipeline",
    "PreprocessConfig",
    "DeviceType",
    "create_detection_pipeline",
    "create_classification_pipeline",
    "create_ocr_pipeline",
    "compute_frame_diff",
    "get_motion_regions",
    # OCR
    "OCRProcessor",
    "OCRConfig",
    "OCREngine",
    "HUDData",
    "HUDParser",
    "create_ocr_processor",
    # Detection
    "DetectionConfig",
    "Detection",
    "DetectionResult",
    "ModelType",
    "create_detector",
    "draw_detections",
    "BaseDetector",
    "YOLODetector",
    "RTDETRDetector",
]

# Version
__version__ = "1.0.0"