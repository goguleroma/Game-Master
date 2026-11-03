"""
Object detection using YOLO/RT-DETR ONNX models with ONNX Runtime DirectML.

Optimized for AMD RX 6800 XT via DirectML execution provider.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    ort = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ModelType(Enum):
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    RT_DETR = "rt_detr"
    CUSTOM = "custom"


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    model_path: str
    model_type: ModelType = ModelType.YOLOV8
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[str]] = None  # Filter by class names
    class_ids: Optional[List[int]] = None  # Filter by class IDs
    use_directml: bool = True
    device_id: int = 0
    half_precision: bool = True
    # Preprocessing
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    swap_rb: bool = True  # BGR to RGB


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h (normalized 0-1)
    bbox_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)


@dataclass
class DetectionResult:
    """Complete detection result for a frame."""
    detections: List[Detection]
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    input_shape: Tuple[int, int]
    frame_id: int


class BaseDetector:
    """Base class for object detectors."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._initialized = False
        self._class_names: List[str] = config.classes or []
    
    def initialize(self) -> bool:
        raise NotImplementedError
    
    def detect(self, image: np.ndarray, frame_id: int = 0) -> DetectionResult:
        raise NotImplementedError
    
    def shutdown(self) -> None:
        self._session = None
        self._initialized = False
    
    @property
    def class_names(self) -> List[str]:
        return self._class_names
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


class YOLODetector(BaseDetector):
    """YOLOv8/v9/v10 detector using ONNX Runtime."""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._input_shape = config.input_size
    
    def initialize(self) -> bool:
        if not ORT_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        if not Path(self.config.model_path).exists():
            logger.error(f"Model not found: {self.config.model_path}")
            return False
        
        try:
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            
            # Execution providers
            providers = []
            if self.config.use_directml:
                try:
                    providers.append(("DmlExecutionProvider", {"device_id": self.config.device_id}))
                    logger.info("Using DirectML execution provider")
                except Exception as e:
                    logger.warning(f"DirectML not available: {e}")
            
            providers.append("CPUExecutionProvider")
            
            # Create session
            self._session = ort.InferenceSession(
                self.config.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            
            # Get input/output info
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            # Try to load class names from model metadata
            self._load_class_names()
            
            # Warmup
            self._warmup()
            
            self._initialized = True
            logger.info(f"YOLO detector initialized: {self.config.model_path}")
            logger.info(f"Input: {self._session.get_inputs()[0].shape}, Outputs: {len(self._output_names)}")
            return True
            
        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
            return False
    
    def _load_class_names(self) -> None:
        """Load class names from model metadata or config."""
        if self.config.classes:
            self._class_names = self.config.classes
            return
        
        # Try to get from model metadata
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import json
                self._class_names = json.loads(meta["names"])
                logger.info(f"Loaded {len(self._class_names)} classes from model metadata")
        except Exception:
            pass
        
        # Default COCO classes if none found
        if not self._class_names:
            self._class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush",
            ]
    
    def _warmup(self) -> None:
        """Warmup the model with dummy input."""
        dummy = np.random.randint(0, 255, (1, 3, *self._input_shape), dtype=np.uint8)
        dummy = dummy.astype(np.float32) / 255.0
        _ = self._session.run(self._output_names, {self._input_name: dummy})
    
    def detect(self, image: np.ndarray, frame_id: int = 0) -> DetectionResult:
        if not self._initialized:
            raise RuntimeError("Detector not initialized")
        
        # Preprocess
        preprocess_start = time.perf_counter()
        input_tensor, scale_info = self._preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Inference
        infer_start = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        infer_time = (time.perf_counter() - infer_start) * 1000
        
        # Postprocess
        post_start = time.perf_counter()
        detections = self._postprocess(outputs[0], scale_info)
        post_time = (time.perf_counter() - post_start) * 1000
        
        return DetectionResult(
            detections=detections,
            inference_time_ms=infer_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=post_time,
            input_shape=image.shape[:2][::-1],  # (w, h)
            frame_id=frame_id,
        )
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Preprocess image for YOLO."""
        h, w = image.shape[:2]
        input_h, input_w = self._input_shape
        
        # Letterbox resize to maintain aspect ratio
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if CV2_AVAILABLE:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            img = Image.fromarray(image)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized = np.array(img)
        
        # Pad to input size
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)  # YOLO uses 114 as pad value
        pad_top = (input_h - new_h) // 2
        pad_left = (input_w - new_w) // 2
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        # BGR to RGB
        if self.config.swap_rb:
            padded = padded[:, :, ::-1]
        
        # Normalize and convert to CHW
        if self.config.normalize:
            padded = padded.astype(np.float32) / 255.0
            mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(self.config.std, dtype=np.float32).reshape(1, 1, 3)
            padded = (padded - mean) / std
        else:
            padded = padded.astype(np.float32)
        
        # HWC to CHW
        padded = padded.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = padded[np.newaxis, ...]
        
        if self.config.half_precision:
            input_tensor = input_tensor.astype(np.float16)
        
        scale_info = {
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
            "orig_w": w,
            "orig_h": h,
            "input_w": input_w,
            "input_h": input_h,
        }
        
        return input_tensor, scale_info
    
    def _postprocess(self, output: np.ndarray, scale_info: Dict) -> List[Detection]:
        """Postprocess YOLO output."""
        # Output shape: (1, num_boxes, 85) or (1, 84, num_boxes) depending on version
        # YOLOv8: (1, 84, 8400) -> transpose to (1, 8400, 84)
        # YOLOv9/v10 similar
        
        if output.ndim == 3:
            if output.shape[1] > output.shape[2]:
                # (1, 84, 8400) -> (1, 8400, 84)
                output = output.transpose(0, 2, 1)
        
        # Squeeze batch
        predictions = output[0]  # (num_boxes, 84) or (num_boxes, 85)
        
        # Extract boxes and scores
        # Format: [x, y, w, h, class_scores...] or [x, y, w, h, obj_conf, class_scores...]
        num_classes = predictions.shape[1] - 4
        
        if num_classes == 80:  # COCO
            # YOLOv8 format: [x, y, w, h, class_0, class_1, ...]
            box_xywh = predictions[:, :4]
            class_scores = predictions[:, 4:]
        elif num_classes == 81:  # With objectness
            # Some versions: [x, y, w, h, obj_conf, class_0, ...]
            box_xywh = predictions[:, :4]
            class_scores = predictions[:, 5:]
        else:
            # Generic: assume first 4 are box, rest are class scores
            box_xywh = predictions[:, :4]
            class_scores = predictions[:, 4:]
        
        # Get max class score and class id
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence
        mask = max_scores >= self.config.confidence_threshold
        if self.config.class_ids:
            class_mask = np.isin(class_ids, self.config.class_ids)
            mask = mask & class_mask
        
        box_xywh = box_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        if len(box_xywh) == 0:
            return []
        
        # Convert xywh (center) to xyxy (corners) in input space
        x_center, y_center, width, height = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale back to original image coordinates
        scale = scale_info["scale"]
        pad_left = scale_info["pad_left"]
        pad_top = scale_info["pad_top"]
        orig_w = scale_info["orig_w"]
        orig_h = scale_info["orig_h"]
        
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Convert to normalized coordinates
        x1_norm = x1 / orig_w
        y1_norm = y1 / orig_h
        x2_norm = x2 / orig_w
        y2_norm = y2 / orig_h
        
        w_norm = (x2 - x1) / orig_w
        h_norm = (y2 - y1) / orig_h
        
        # NMS
        indices = self._nms(
            np.stack([x1, y1, x2, y2], axis=1),
            max_scores,
            self.config.iou_threshold,
        )
        
        # Limit detections
        indices = indices[:self.config.max_detections]
        
        # Build detections
        detections = []
        for idx in indices:
            class_id = int(class_ids[idx])
            class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            
            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(max_scores[idx]),
                bbox=(float(x1_norm[idx]), float(y1_norm[idx]), float(w_norm[idx]), float(h_norm[idx])),
                bbox_xyxy=(float(x1_norm[idx]), float(y1_norm[idx]), float(x2_norm[idx]), float(y2_norm[idx])),
            ))
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area_others - inter + 1e-6)
            
            # Keep boxes with IoU < threshold
            order = order[1:][iou < iou_threshold]
        
        return np.array(keep, dtype=np.int32)


class RTDETRDetector(BaseDetector):
    """RT-DETR detector using ONNX Runtime."""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        # RT-DETR typically uses fixed input size
        self._input_shape = config.input_size
    
    def initialize(self) -> bool:
        if not ORT_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        if not Path(self.config.model_path).exists():
            logger.error(f"Model not found: {self.config.model_path}")
            return False
        
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = []
            if self.config.use_directml:
                try:
                    providers.append(("DmlExecutionProvider", {"device_id": self.config.device_id}))
                except Exception:
                    pass
            providers.append("CPUExecutionProvider")
            
            self._session = ort.InferenceSession(
                self.config.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self._load_class_names()
            self._warmup()
            
            self._initialized = True
            logger.info(f"RT-DETR detector initialized: {self.config.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"RT-DETR initialization failed: {e}")
            return False
    
    def _load_class_names(self) -> None:
        if self.config.classes:
            self._class_names = self.config.classes
        else:
            # COCO classes
            self._class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush",
            ]
    
    def _warmup(self) -> None:
        dummy = np.random.randint(0, 255, (1, 3, *self._input_shape), dtype=np.uint8)
        dummy = dummy.astype(np.float32) / 255.0
        _ = self._session.run(self._output_names, {self._input_name: dummy})
    
    def detect(self, image: np.ndarray, frame_id: int = 0) -> DetectionResult:
        if not self._initialized:
            raise RuntimeError("Detector not initialized")
        
        preprocess_start = time.perf_counter()
        input_tensor = self._preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        infer_start = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        infer_time = (time.perf_counter() - infer_start) * 1000
        
        post_start = time.perf_counter()
        detections = self._postprocess(outputs, image.shape[:2])
        post_time = (time.perf_counter() - post_start) * 1000
        
        return DetectionResult(
            detections=detections,
            inference_time_ms=infer_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=post_time,
            input_shape=image.shape[:2][::-1],
            frame_id=frame_id,
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        input_h, input_w = self._input_shape
        
        if CV2_AVAILABLE:
            resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            img = Image.fromarray(image)
            img = img.resize((input_w, input_h), Image.Resampling.BILINEAR)
            resized = np.array(img)
        
        if self.config.swap_rb:
            resized = resized[:, :, ::-1]
        
        if self.config.normalize:
            resized = resized.astype(np.float32) / 255.0
            mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(self.config.std, dtype=np.float32).reshape(1, 1, 3)
            resized = (resized - mean) / std
        else:
            resized = resized.astype(np.float32)
        
        resized = resized.transpose(2, 0, 1)
        input_tensor = resized[np.newaxis, ...]
        
        if self.config.half_precision:
            input_tensor = input_tensor.astype(np.float16)
        
        return input_tensor
    
    def _postprocess(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[Detection]:
        # RT-DETR outputs: [labels, boxes, scores]
        # shapes: labels (1, num_queries), boxes (1, num_queries, 4), scores (1, num_queries)
        
        if len(outputs) < 3:
            logger.warning(f"Unexpected RT-DETR outputs: {len(outputs)}")
            return []
        
        labels = outputs[0][0]  # (num_queries,)
        boxes = outputs[1][0]   # (num_queries, 4) - normalized cxcywh
        scores = outputs[2][0]  # (num_queries,)
        
        # Filter by confidence
        mask = scores >= self.config.confidence_threshold
        if self.config.class_ids:
            class_mask = np.isin(labels, self.config.class_ids)
            mask = mask & class_mask
        
        labels = labels[mask].astype(int)
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert cxcywh to xyxy (normalized)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Clip
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)
        
        # NMS per class
        keep_indices = []
        for class_id in np.unique(labels):
            class_mask = labels == class_id
            class_boxes = np.stack([x1[class_mask], y1[class_mask], x2[class_mask], y2[class_mask]], axis=1)
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]
            
            nms_keep = self._nms(class_boxes, class_scores, self.config.iou_threshold)
            keep_indices.extend(class_indices[nms_keep])
        
        keep_indices = np.array(keep_indices)
        keep_indices = keep_indices[:self.config.max_detections]
        
        # Build detections
        detections = []
        for idx in keep_indices:
            class_id = int(labels[idx])
            class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            
            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(scores[idx]),
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx] - x1[idx]), float(y2[idx] - y1[idx])),
                bbox_xyxy=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
            ))
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area_others - inter + 1e-6)
            order = order[1:][iou < iou_threshold]
        
        return np.array(keep, dtype=np.int32)


def create_detector(config: DetectionConfig) -> BaseDetector:
    """Factory function to create detector based on model type."""
    if config.model_type in (ModelType.YOLOV8, ModelType.YOLOV9, ModelType.YOLOV10):
        return YOLODetector(config)
    elif config.model_type == ModelType.RT_DETR:
        return RTDETRDetector(config)
    else:
        # Try to infer from model name
        model_name = Path(config.model_path).stem.lower()
        if "rt_detr" in model_name or "rtdetr" in model_name:
            return RTDETRDetector(config)
        return YOLODetector(config)


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw detection boxes on image."""
    if cv2 is None:
        return image
    
    output = image.copy()
    h, w = image.shape[:2]
    
    for det in detections:
        # Convert normalized to pixel coordinates
        x1 = int(det.bbox_xyxy[0] * w)
        y1 = int(det.bbox_xyxy[1] * h)
        x2 = int(det.bbox_xyxy[2] * w)
        y2 = int(det.bbox_xyxy[3] * h)
        
        # Color
        if class_colors and det.class_id in class_colors:
            color = class_colors[det.class_id]
        else:
            # Generate consistent color from class_id
            np.random.seed(det.class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(output, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
        cv2.putText(output, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print(f"ONNX Runtime available: {ORT_AVAILABLE}")
    if ORT_AVAILABLE:
        print(f"Available providers: {ort.get_available_providers()}")
    
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # Test detector creation (without model)
    config = DetectionConfig(
        model_path="models/yolov8n.onnx",
        model_type=ModelType.YOLOV8,
        confidence_threshold=0.25,
    )
    detector = create_detector(config)
    print(f"Created detector: {type(detector).__name__}")