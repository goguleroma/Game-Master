"""
Real screen capture implementation for Windows using DirectX (dxcam) with mss fallback.

Target: 30fps sustained @ 1920x1080 on AMD RX 6800 XT with <16ms frame-to-frame latency.
"""

import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union, List, Deque
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for screen capture."""
    target_fps: int = 30
    width: int = 1920
    height: int = 1080
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) or None for full screen
    use_gpu_preprocess: bool = True
    ring_buffer_size: int = 30  # Store last N frames for frame differencing
    output_color: str = "BGR"  # "BGR" (OpenCV), "RGB" (PIL/torch)


@dataclass
class FrameMetrics:
    """Metrics for captured frame."""
    frame_id: int
    timestamp: float
    capture_latency_ms: float
    process_latency_ms: float
    total_latency_ms: float
    fps: float


class FrameRingBuffer:
    """Thread-safe ring buffer for frame storage and differencing."""
    
    def __init__(self, max_size: int = 30):
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_size)
        self._timestamps: Deque[float] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._frame_ids: Deque[int] = deque(maxlen=max_size)
        self._next_frame_id = 0
    
    def append(self, frame: np.ndarray, timestamp: float) -> int:
        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            self._buffer.append(frame.copy())
            self._timestamps.append(timestamp)
            self._frame_ids.append(frame_id)
            return frame_id
    
    def get_latest(self) -> Optional[Tuple[np.ndarray, float, int]]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1], self._timestamps[-1], self._frame_ids[-1]
    
    def get_previous(self, n: int = 1) -> Optional[Tuple[np.ndarray, float, int]]:
        with self._lock:
            if len(self._buffer) <= n:
                return None
            idx = -(n + 1)
            return self._buffer[idx], self._timestamps[idx], self._frame_ids[idx]
    
    def get_frame_diff(self, n: int = 1) -> Optional[np.ndarray]:
        """Get absolute difference between latest frame and n frames ago."""
        latest = self.get_latest()
        previous = self.get_previous(n)
        if latest is None or previous is None:
            return None
        latest_frame = latest[0].astype(np.float32)
        prev_frame = previous[0].astype(np.float32)
        diff = np.abs(latest_frame - prev_frame)
        return diff.astype(latest_frame.dtype)
    
    def get_changed_regions(self, n: int = 1, threshold: float = 10.0) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes of changed regions between frames."""
        diff = self.get_frame_diff(n)
        if diff is None:
            return []
        
        # Convert to grayscale if needed
        if diff.ndim == 3:
            gray_diff = np.mean(diff, axis=2)
        else:
            gray_diff = diff
        
        # Threshold
        _, binary = cv2.threshold(gray_diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter noise
                regions.append((x, y, w, h))
        
        return regions
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()
            self._frame_ids.clear()


# Try to import optional dependencies
try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    dxcam = None

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    mss = None

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


class ScreenCapture:
    """
    High-performance screen capture using DirectX (dxcam) on Windows,
    with mss as cross-platform fallback.
    
    Implements the ObservationAdapter protocol for integration with GameEngine.
    """
    
    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        device_idx: int = 0,
        output_idx: int = 0,
    ) -> None:
        """
        Initialize screen capture.
        
        Args:
            config: Capture configuration
            device_idx: GPU device index for dxcam
            output_idx: Monitor/output index for dxcam
        """
        self.config = config or CaptureConfig()
        self._device_idx = device_idx
        self._output_idx = output_idx
        
        # Capture backend
        self._camera = None
        self._mss_instance = None
        self._backend = None  # "dxcam" or "mss"
        
        # Threading
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        
        # Frame storage
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_lock = threading.RLock()
        
        # Ring buffer for frame history
        self._ring_buffer = FrameRingBuffer(self.config.ring_buffer_size)
        
        # ROI / Attention region
        self._attention_region: Optional[Tuple[int, int, int, int]] = None
        self._roi_lock = threading.RLock()
        
        # Metrics
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        self._metrics_history: List[FrameMetrics] = []
        self._metrics_lock = threading.RLock()
        
        # Benchmarking hooks
        self._benchmark_callback: Optional[callable] = None
        
        logger.info(f"ScreenCapture initialized: {self.config.width}x{self.config.height} @ {self.config.target_fps}fps")
    
    @property
    def backend(self) -> str:
        return self._backend or "unknown"
    
    @property
    def is_running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()
    
    @property
    def current_fps(self) -> float:
        return self._current_fps
    
    @property
    def attention_region(self) -> Optional[Tuple[int, int, int, int]]:
        with self._roi_lock:
            return self._attention_region
    
    def set_attention_region(self, x: int, y: int, w: int, h: int) -> None:
        """Set ROI for attention-focused capture."""
        with self._roi_lock:
            self._attention_region = (x, y, w, h)
            logger.info(f"Attention region set: x={x}, y={y}, w={w}, h={h}")
    
    def clear_attention_region(self) -> None:
        """Clear ROI, capture full region."""
        with self._roi_lock:
            self._attention_region = None
            logger.info("Attention region cleared")
    
    def set_benchmark_callback(self, callback: callable) -> None:
        """Set callback for benchmarking: callback(metrics: FrameMetrics)."""
        self._benchmark_callback = callback
    
    def start(self) -> bool:
        """Start capture thread."""
        if self.is_running:
            logger.warning("Capture already running")
            return True
        
        # Initialize backend
        if not self._init_backend():
            logger.error("Failed to initialize any capture backend")
            return False
        
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        # Wait for first frame
        if not self._frame_ready.wait(timeout=5.0):
            logger.error("Timeout waiting for first frame")
            self.stop()
            return False
        
        logger.info(f"Capture started on {self._backend} backend")
        return True
    
    def stop(self) -> None:
        """Stop capture thread."""
        if not self.is_running:
            return
        
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        self._cleanup_backend()
        logger.info("Capture stopped")
    
    def _init_backend(self) -> bool:
        """Initialize capture backend (dxcam preferred, mss fallback)."""
        # Try dxcam first (DirectX, Windows only, best performance)
        if DXCAM_AVAILABLE:
            try:
                self._camera = dxcam.create(
                    device_idx=self._device_idx,
                    output_idx=self._output_idx,
                    output_color=self.config.output_color,
                )
                # Test capture
                test_frame = self._camera.grab()
                if test_frame is not None:
                    self._backend = "dxcam"
                    logger.info("dxcam backend initialized successfully")
                    return True
            except Exception as e:
                logger.warning(f"dxcam initialization failed: {e}")
                self._camera = None
        
        # Fallback to mss (cross-platform, slower)
        if MSS_AVAILABLE:
            try:
                self._mss_instance = mss.mss()
                # Test capture
                monitor = self._mss_instance.monitors[self._output_idx + 1] if self._output_idx + 1 < len(self._mss_instance.monitors) else self._mss_instance.monitors[0]
                test_frame = np.array(self._mss_instance.grab(monitor))
                if test_frame is not None:
                    self._backend = "mss"
                    logger.info("mss backend initialized successfully")
                    return True
            except Exception as e:
                logger.warning(f"mss initialization failed: {e}")
                self._mss_instance = None
        
        return False
    
    def _cleanup_backend(self) -> None:
        """Clean up capture backend resources."""
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        
        if self._mss_instance:
            try:
                self._mss_instance.close()
            except Exception:
                pass
            self._mss_instance = None
        
        self._backend = None
    
    def _get_capture_region(self) -> Tuple[int, int, int, int]:
        """Get the capture region considering attention region."""
        with self._roi_lock:
            if self._attention_region:
                return self._attention_region
        
        if self.config.region:
            return self.config.region
        
        # Default to full configured resolution
        return (0, 0, self.config.width, self.config.height)
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        target_frame_time = 1.0 / self.config.target_fps
        last_frame_time = time.perf_counter()
        
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            
            # Capture frame
            frame = self._grab_frame()
            capture_time = time.perf_counter()
            
            if frame is not None:
                # Process frame (resize, color convert, etc.)
                processed_frame = self._process_frame(frame)
                process_time = time.perf_counter()
                
                # Store frame
                with self._frame_lock:
                    self._latest_frame = processed_frame
                    self._latest_timestamp = capture_time
                
                # Add to ring buffer
                frame_id = self._ring_buffer.append(processed_frame, capture_time)
                
                # Update FPS
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._current_fps = self._frame_count / (now - self._last_fps_time)
                    self._frame_count = 0
                    self._last_fps_time = now
                
                # Record metrics
                metrics = FrameMetrics(
                    frame_id=frame_id,
                    timestamp=capture_time,
                    capture_latency_ms=(capture_time - loop_start) * 1000,
                    process_latency_ms=(process_time - capture_time) * 1000,
                    total_latency_ms=(process_time - loop_start) * 1000,
                    fps=self._current_fps,
                )
                self._record_metrics(metrics)
                
                # Signal frame ready
                self._frame_ready.set()
            
            # Frame rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check for frame drops
            actual_frame_time = time.perf_counter() - last_frame_time
            if actual_frame_time > target_frame_time * 1.5:
                logger.debug(f"Frame time exceeded target: {actual_frame_time*1000:.1f}ms > {target_frame_time*1000:.1f}ms")
            last_frame_time = time.perf_counter()
    
    def _grab_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from the active backend."""
        region = self._get_capture_region()
        x, y, w, h = region
        
        if self._backend == "dxcam" and self._camera:
            try:
                # dxcam uses (left, top, right, bottom)
                frame = self._camera.grab(region=(x, y, x + w, y + h))
                return frame
            except Exception as e:
                logger.error(f"dxcam grab failed: {e}")
                return None
        
        elif self._backend == "mss" and self._mss_instance:
            try:
                monitor = {"left": x, "top": y, "width": w, "height": h}
                frame = np.array(self._mss_instance.grab(monitor))
                # mss returns BGRA, convert to BGR
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                return frame
            except Exception as e:
                logger.error(f"mss grab failed: {e}")
                return None
        
        return None
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: resize, color convert, normalize."""
        # Resize if needed
        target_w, target_h = self.config.width, self.config.height
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            if CV2_AVAILABLE:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback using PIL
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                frame = np.array(img)
        
        # Ensure correct color format
        if self.config.output_color == "RGB" and frame.shape[2] == 3:
            # Assume input is BGR (OpenCV default), convert to RGB
            if CV2_AVAILABLE:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = frame[:, :, ::-1]
        
        return frame
    
    def _record_metrics(self, metrics: FrameMetrics) -> None:
        """Record metrics and invoke callback."""
        with self._metrics_lock:
            self._metrics_history.append(metrics)
            # Keep only last 1000 metrics
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-1000:]
        
        if self._benchmark_callback:
            try:
                self._benchmark_callback(metrics)
            except Exception as e:
                logger.warning(f"Benchmark callback failed: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest captured frame (thread-safe)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None
    
    def get_frame_with_metadata(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get latest frame with timestamp and frame_id."""
        return self._ring_buffer.get_latest()
    
    def get_frame_diff(self, n: int = 1) -> Optional[np.ndarray]:
        """Get frame difference with n frames ago."""
        return self._ring_buffer.get_frame_diff(n)
    
    def get_changed_regions(self, n: int = 1, threshold: float = 10.0) -> List[Tuple[int, int, int, int]]:
        """Get changed regions between frames."""
        return self._ring_buffer.get_changed_regions(n, threshold)
    
    def get_metrics(self, count: int = 100) -> List[FrameMetrics]:
        """Get recent metrics."""
        with self._metrics_lock:
            return self._metrics_history[-count:]
    
    def get_average_latency(self) -> float:
        """Get average total latency in ms."""
        metrics = self.get_metrics(100)
        if not metrics:
            return 0.0
        return sum(m.total_latency_ms for m in metrics) / len(metrics)
    
    def extract(self, observation: Any) -> Any:
        """
        Implement ObservationAdapter protocol.
        Pass-through for frame data.
        """
        from core.types import GameState
        return GameState(
            objects=["captured_frame"],
            hud={"backend": self._backend, "fps": self._current_fps},
            goal="capture",
            confidence=1.0 if self._latest_frame is not None else 0.0,
        )
    
    def __enter__(self) -> "ScreenCapture":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# Convenience function for quick capture
def capture_once(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """Capture a single frame (blocking)."""
    config = CaptureConfig()
    if region:
        config.region = region
    
    with ScreenCapture(config) as cap:
        # Wait for a frame
        for _ in range(10):
            frame = cap.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
    return None


class ScreenCaptureAdapter:
    """Adapter that combines ScreenCapture and StateExtractor into an ObservationAdapter."""

    def __init__(
        self,
        capture: ScreenCapture | None = None,
        state_extractor: "StateExtractor | SimpleStateExtractor | None" = None,
    ) -> None:
        from perception.state_extractor import StateExtractor, SimpleStateExtractor
        self._capture = capture or ScreenCapture()
        self._state_extractor = state_extractor or SimpleStateExtractor()

    def get_frame(self) -> Any | None:
        return self._capture.get_frame()

    def extract(self, observation: Observation) -> GameState:
        return self._state_extractor.extract(observation)

    def reset(self) -> None:
        if hasattr(self._capture, 'stop'):
            self._capture.stop()

    @property
    def is_real_time(self) -> bool:
        return hasattr(self._capture, 'is_running') and self._capture.is_running

    @property
    def frame_shape(self) -> tuple[int, ...]:
        if hasattr(self._capture, 'config'):
            return (self._capture.config.height, self._capture.config.width, 3)
        return (0, 0, 0)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    config = CaptureConfig(target_fps=30, width=1920, height=1080)
    
    def print_metrics(m: FrameMetrics):
        print(f"Frame {m.frame_id}: capture={m.capture_latency_ms:.1f}ms, "
              f"process={m.process_latency_ms:.1f}ms, total={m.total_latency_ms:.1f}ms, "
              f"fps={m.fps:.1f}")
    
    with ScreenCapture(config) as capture:
        capture.set_benchmark_callback(print_metrics)
        time.sleep(3)
        print(f"Average latency: {capture.get_average_latency():.1f}ms")