"""
GPU-accelerated preprocessing pipeline using torch-directml for AMD RX 6800 XT.

Provides: resize, normalize, color conversion, tensor conversion for neural networks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Union
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch and directml
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False
    torch_directml = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    AUTO = "auto"


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    target_size: Tuple[int, int] = (224, 224)  # (width, height)
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet default
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    to_tensor: bool = True
    to_rgb: bool = True  # Convert BGR to RGB
    device: DeviceType = DeviceType.AUTO
    half_precision: bool = False  # Use float16
    batch_dim: bool = True  # Add batch dimension


class PreprocessingPipeline:
    """
    GPU-accelerated preprocessing pipeline for game frames.
    
    Uses torch-directml for AMD GPU acceleration on Windows.
    Falls back to CPU/OpenCV if GPU unavailable.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self._device = self._select_device()
        self._torch_device = self._get_torch_device()
        self._mean_tensor = None
        self._std_tensor = None
        self._initialized = False
        
        logger.info(f"PreprocessingPipeline initialized on {self._device.value}")
    
    def _select_device(self) -> DeviceType:
        """Select best available device."""
        if self.config.device != DeviceType.AUTO:
            return self.config.device
        
        if TORCH_AVAILABLE:
            if DIRECTML_AVAILABLE:
                try:
                    # Test DirectML
                    dml_device = torch_directml.device()
                    _ = torch.tensor([1.0], device=dml_device)
                    return DeviceType.DIRECTML
                except Exception as e:
                    logger.warning(f"DirectML not available: {e}")
            
            if torch.cuda.is_available():
                return DeviceType.CUDA
        
        return DeviceType.CPU
    
    def _is_torch_available(self) -> bool:
        """Check if torch is available (can be patched for testing)."""
        return TORCH_AVAILABLE
    
    def _get_torch_device(self) -> "torch.device":
        """Get torch device object."""
        if not self._is_torch_available():
            # Return a mock device object that stringifies to "cpu"
            class MockDevice:
                def __str__(self):
                    return "cpu"
            return MockDevice()
        if self._device == DeviceType.DIRECTML:
            return torch_directml.device()
        elif self._device == DeviceType.CUDA:
            return torch.device("cuda")
        # CPU device - use mock if torch not available
        if not self._is_torch_available():
            class MockDevice:
                def __str__(self):
                    return "cpu"
            return MockDevice()
        return torch.device("cpu")
    
    def _lazy_init(self) -> None:
        """Lazy initialization of tensors on device."""
        if self._initialized:
            return
        
        if self._is_torch_available() and self.config.normalize:
            self._mean_tensor = torch.tensor(
                self.config.mean, device=self._torch_device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            self._std_tensor = torch.tensor(
                self.config.std, device=self._torch_device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            
            if self.config.half_precision:
                self._mean_tensor = self._mean_tensor.half()
                self._std_tensor = self._std_tensor.half()
        
        self._initialized = True
    
    def process(self, frame: np.ndarray) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR or RGB
            
        Returns:
            Processed frame as tensor (1, C, H, W) or numpy array
        """
        self._lazy_init()
        
        # Use GPU path only when on non-CPU device and torch is available
        # CPU device always uses CPU path regardless of to_tensor setting
        if TORCH_AVAILABLE and self._device != DeviceType.CPU:
            return self._process_gpu(frame)
        else:
            return self._process_cpu(frame)
    
    def _process_gpu(self, frame: np.ndarray) -> "torch.Tensor":
        """Process frame on GPU using torch-directml."""
        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(frame).to(self._torch_device)
        
        # Handle dtype
        if self.config.half_precision:
            tensor = tensor.half()
        else:
            tensor = tensor.float()
        
        # BGR to RGB if needed
        if self.config.to_rgb and tensor.shape[2] == 3:
            tensor = tensor.flip(dims=[2])
        
        # Permute to (C, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Resize if needed
        target_h, target_w = self.config.target_size[1], self.config.target_size[0]
        if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        
        # Normalize to [0, 1] then standardize
        tensor = tensor / 255.0
        
        if self.config.normalize:
            tensor = (tensor - self._mean_tensor) / self._std_tensor
        
        if not self.config.batch_dim:
            tensor = tensor.squeeze(0)
        
        return tensor
    
    def _process_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Process frame on CPU using OpenCV."""
        processed = frame.copy()
        
        # BGR to RGB
        if self.config.to_rgb and CV2_AVAILABLE and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        elif self.config.to_rgb:
            processed = processed[:, :, ::-1]
        
        # Resize
        target_w, target_h = self.config.target_size
        if processed.shape[1] != target_w or processed.shape[0] != target_h:
            if CV2_AVAILABLE:
                processed = cv2.resize(
                    processed, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                from PIL import Image
                img = Image.fromarray(processed)
                img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                processed = np.array(img)
        
        # Normalize to [0, 1]
        if self.config.normalize:
            processed = processed.astype(np.float32) / 255.0
            mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(self.config.std, dtype=np.float32).reshape(1, 1, 3)
            processed = (processed - mean) / std
        
        # Add batch dim and channel-first if tensor output requested
        if self.config.to_tensor:
            processed = processed.transpose(2, 0, 1)  # (C, H, W)
            if self.config.batch_dim:
                processed = processed[np.newaxis, ...]  # (1, C, H, W)
        
        return processed
    
    def process_batch(self, frames: List[np.ndarray]) -> Union[np.ndarray, "torch.Tensor"]:
        """Process multiple frames as a batch."""
        if not frames:
            raise ValueError("Empty frame list")
        
        # Use GPU path only when on non-CPU device and torch is available
        if TORCH_AVAILABLE and self._device != DeviceType.CPU:
            return self._process_batch_gpu(frames)
        else:
            return self._process_batch_cpu(frames)
    
    def _process_batch_gpu(self, frames: List[np.ndarray]) -> "torch.Tensor":
        """Process batch on GPU."""
        self._lazy_init()
        
        # Stack frames
        tensors = []
        for frame in frames:
            tensor = torch.from_numpy(frame).to(self._torch_device)
            if self.config.half_precision:
                tensor = tensor.half()
            else:
                tensor = tensor.float()
            
            if self.config.to_rgb and tensor.shape[2] == 3:
                tensor = tensor.flip(dims=[2])
            
            tensor = tensor.permute(2, 0, 1)
            tensors.append(tensor)
        
        batch = torch.stack(tensors, dim=0)  # (B, C, H, W)
        
        # Resize
        target_h, target_w = self.config.target_size[1], self.config.target_size[0]
        if batch.shape[2] != target_h or batch.shape[3] != target_w:
            batch = torch.nn.functional.interpolate(
                batch,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        
        # Normalize
        batch = batch / 255.0
        if self.config.normalize:
            batch = (batch - self._mean_tensor) / self._std_tensor
        
        return batch
    
    def _process_batch_cpu(self, frames: List[np.ndarray]) -> np.ndarray:
        """Process batch on CPU."""
        # Temporarily disable batch_dim to avoid double batch dimension
        original_batch_dim = self.config.batch_dim
        self.config.batch_dim = False
        try:
            processed = [self._process_cpu(f) for f in frames]
        finally:
            self.config.batch_dim = original_batch_dim
        return np.stack(processed, axis=0)
    
    @property
    def device(self) -> DeviceType:
        return self._device
    
    @property
    def torch_device(self) -> "torch.device":
        return self._torch_device


# Specialized pipelines for common use cases
def create_detection_pipeline(
    input_size: Tuple[int, int] = (640, 640),
    device: DeviceType = DeviceType.AUTO,
) -> PreprocessingPipeline:
    """Create pipeline optimized for YOLO/RT-DETR detection."""
    return PreprocessingPipeline(PreprocessConfig(
        target_size=input_size,
        normalize=True,
        mean=(0.0, 0.0, 0.0),  # YOLO expects 0-1 range
        std=(1.0, 1.0, 1.0),
        to_rgb=True,
        device=device,
        half_precision=True,
    ))


def create_classification_pipeline(
    input_size: Tuple[int, int] = (224, 224),
    device: DeviceType = DeviceType.AUTO,
) -> PreprocessingPipeline:
    """Create pipeline optimized for classification (ImageNet-style)."""
    return PreprocessingPipeline(PreprocessConfig(
        target_size=input_size,
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        to_rgb=True,
        device=device,
        half_precision=False,
    ))


def create_ocr_pipeline(
    input_size: Tuple[int, int] = (320, 320),
    device: DeviceType = DeviceType.AUTO,
) -> PreprocessingPipeline:
    """Create pipeline optimized for OCR (PaddleOCR)."""
    return PreprocessingPipeline(PreprocessConfig(
        target_size=input_size,
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        to_rgb=True,
        device=device,
        half_precision=False,
    ))


# Frame differencing utilities
def compute_frame_diff(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "abs",
    threshold: float = 10.0,
) -> np.ndarray:
    """
    Compute difference between two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        method: "abs" (absolute), "squared", "relative"
        threshold: Threshold for binary mask
        
    Returns:
        Difference map or binary mask
    """
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes differ: {frame1.shape} vs {frame2.shape}")
    
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    if method == "abs":
        diff = np.abs(f1 - f2)
    elif method == "squared":
        diff = (f1 - f2) ** 2
    elif method == "relative":
        diff = np.abs(f1 - f2) / (np.maximum(f1, f2) + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to grayscale if color
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
    
    # Threshold
    if threshold > 0:
        diff = (diff > threshold).astype(np.float32)
    
    return diff


def get_motion_regions(
    frame1: np.ndarray,
    frame2: np.ndarray,
    min_area: int = 100,
    threshold: float = 15.0,
) -> List[Tuple[int, int, int, int]]:
    """Get bounding boxes of motion regions between frames."""
    if not CV2_AVAILABLE or cv2 is None:
        logger.warning("OpenCV not available for motion detection")
        return []
    
    diff = compute_frame_diff(frame1, frame2, threshold=threshold)
    diff_uint8 = (diff * 255).astype(np.uint8)
    
    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_uint8 = cv2.morphologyEx(diff_uint8, cv2.MORPH_OPEN, kernel)
    diff_uint8 = cv2.morphologyEx(diff_uint8, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(diff_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
    
    return regions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test pipeline
    pipeline = create_detection_pipeline()
    print(f"Device: {pipeline.device}")
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    import time
    start = time.time()
    for _ in range(10):
        result = pipeline.process(dummy_frame)
    elapsed = time.time() - start
    print(f"10 frames processed in {elapsed*1000:.1f}ms ({10/elapsed:.1f} fps)")
    print(f"Output shape: {result.shape}, dtype: {result.dtype}")