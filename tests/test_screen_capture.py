"""Tests for screen capture functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, Tuple, List

# Import the capture module
from perception.capture import (
    ScreenCapture,
    CaptureConfig,
    FrameMetrics,
    FrameRingBuffer,
    ScreenCaptureAdapter,
    capture_once,
)


class TestCaptureConfig:
    """Tests for CaptureConfig dataclass."""

    def test_default_config(self) -> None:
        config = CaptureConfig()
        assert config.target_fps == 30
        assert config.width == 1920
        assert config.height == 1080
        assert config.region is None
        assert config.use_gpu_preprocess is True
        assert config.ring_buffer_size == 30
        assert config.output_color == "BGR"

    def test_custom_config(self) -> None:
        config = CaptureConfig(
            target_fps=60,
            width=1280,
            height=720,
            region=(100, 100, 800, 600),
            use_gpu_preprocess=False,
            ring_buffer_size=60,
            output_color="RGB",
        )
        assert config.target_fps == 60
        assert config.width == 1280
        assert config.height == 720
        assert config.region == (100, 100, 800, 600)
        assert config.use_gpu_preprocess is False
        assert config.ring_buffer_size == 60
        assert config.output_color == "RGB"


class TestFrameMetrics:
    """Tests for FrameMetrics dataclass."""

    def test_frame_metrics_creation(self) -> None:
        metrics = FrameMetrics(
            frame_id=1,
            timestamp=1234567890.0,
            capture_latency_ms=5.0,
            process_latency_ms=3.0,
            total_latency_ms=8.0,
            fps=30.0,
        )
        assert metrics.frame_id == 1
        assert metrics.timestamp == 1234567890.0
        assert metrics.capture_latency_ms == 5.0
        assert metrics.process_latency_ms == 3.0
        assert metrics.total_latency_ms == 8.0
        assert metrics.fps == 30.0


class TestFrameRingBuffer:
    """Tests for FrameRingBuffer."""

    def test_append_and_get_latest(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        id1 = buffer.append(frame1, 1000.0)
        id2 = buffer.append(frame2, 1001.0)

        assert id1 == 0
        assert id2 == 1

        latest = buffer.get_latest()
        assert latest is not None
        frame, timestamp, frame_id = latest
        assert frame_id == 1
        assert timestamp == 1001.0
        np.testing.assert_array_equal(frame, frame2)

    def test_get_previous(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        for i, frame in enumerate(frames):
            buffer.append(frame, float(1000 + i))

        # Get previous (1 frame back)
        prev = buffer.get_previous(1)
        assert prev is not None
        frame, timestamp, frame_id = prev
        assert frame_id == 1
        np.testing.assert_array_equal(frame, frames[1])

    def test_get_previous_out_of_bounds(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        buffer.append(frame, 1000.0)

        # Request more than available
        prev = buffer.get_previous(5)
        assert prev is None

    def test_get_frame_diff(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frame1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 150, dtype=np.uint8)

        buffer.append(frame1, 1000.0)
        buffer.append(frame2, 1001.0)

        diff = buffer.get_frame_diff(1)
        assert diff is not None
        # Difference should be 50 (150 - 100)
        assert np.all(diff == 50)

    def test_get_frame_diff_not_enough_frames(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        buffer.append(frame, 1000.0)

        diff = buffer.get_frame_diff(1)
        assert diff is None

    def test_get_changed_regions(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        # Create two frames with a known difference region
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[20:40, 30:60] = 255  # White rectangle

        buffer.append(frame1, 1000.0)
        buffer.append(frame2, 1001.0)

        regions = buffer.get_changed_regions(1, threshold=10.0)
        assert len(regions) > 0
        # Should detect the changed region
        x, y, w, h = regions[0]
        assert x <= 30
        assert y <= 20
        assert x + w >= 60
        assert y + h >= 40

    def test_get_changed_regions_no_change(self) -> None:
        buffer = FrameRingBuffer(max_size=5)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        buffer.append(frame, 1000.0)
        buffer.append(frame.copy(), 1001.0)

        regions = buffer.get_changed_regions(1, threshold=10.0)
        assert len(regions) == 0

    def test_buffer_max_size(self) -> None:
        buffer = FrameRingBuffer(max_size=3)
        for i in range(5):
            frame = np.full((10, 10, 3), i, dtype=np.uint8)
            buffer.append(frame, float(i))

        assert len(buffer) == 3
        # Should only have last 3 frames
        latest = buffer.get_latest()
        assert latest[2] == 4  # frame_id 4


class TestScreenCapture:
    """Tests for ScreenCapture class."""

    def test_init_default_config(self) -> None:
        capture = ScreenCapture()
        assert capture.config.target_fps == 30
        assert capture.config.width == 1920
        assert capture.config.height == 1080
        assert capture.backend == "unknown"
        assert capture.is_running is False

    def test_init_custom_config(self) -> None:
        config = CaptureConfig(target_fps=60, width=1280, height=720)
        capture = ScreenCapture(config)
        assert capture.config.target_fps == 60
        assert capture.config.width == 1280
        assert capture.config.height == 720

    def test_attention_region(self) -> None:
        capture = ScreenCapture()
        assert capture.attention_region is None

        capture.set_attention_region(100, 100, 500, 400)
        assert capture.attention_region == (100, 100, 500, 400)

        capture.clear_attention_region()
        assert capture.attention_region is None

    def test_benchmark_callback(self) -> None:
        capture = ScreenCapture()
        metrics_received: List[FrameMetrics] = []

        def callback(metrics: FrameMetrics) -> None:
            metrics_received.append(metrics)

        capture.set_benchmark_callback(callback)
        assert capture._benchmark_callback == callback

    def test_get_frame_returns_none_when_not_started(self) -> None:
        capture = ScreenCapture()
        frame = capture.get_frame()
        assert frame is None

    def test_get_frame_with_metadata_returns_none_when_empty(self) -> None:
        capture = ScreenCapture()
        result = capture.get_frame_with_metadata()
        assert result is None

    def test_get_frame_diff_returns_none_when_empty(self) -> None:
        capture = ScreenCapture()
        diff = capture.get_frame_diff(1)
        assert diff is None

    def test_get_changed_regions_returns_empty_when_empty(self) -> None:
        capture = ScreenCapture()
        regions = capture.get_changed_regions(1)
        assert regions == []

    def test_get_metrics_returns_empty_when_no_data(self) -> None:
        capture = ScreenCapture()
        metrics = capture.get_metrics()
        assert metrics == []

    def test_get_average_latency_returns_zero_when_no_data(self) -> None:
        capture = ScreenCapture()
        avg = capture.get_average_latency()
        assert avg == 0.0

    def test_extract_method(self) -> None:
        capture = ScreenCapture()
        # Without frame
        state = capture.extract({})
        assert isinstance(state, object)  # Returns GameState
        assert hasattr(state, 'objects')
        assert hasattr(state, 'confidence')
        assert state.confidence == 0.0

        # With frame
        capture._latest_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = capture.extract({})
        assert state.confidence == 1.0

    def test_context_manager(self) -> None:
        capture = ScreenCapture()
        # Test that context manager methods exist
        assert hasattr(capture, '__enter__')
        assert hasattr(capture, '__exit__')


class TestScreenCaptureWithMockedBackend:
    """Tests for ScreenCapture with mocked backends."""

    @patch('perception.capture.dxcam', None)
    @patch('perception.capture.mss', None)
    def test_init_backend_none_available(self) -> None:
        """Test initialization when no backend is available."""
        capture = ScreenCapture()
        # Manually set backend to None to simulate failure
        capture._backend = None
        capture._camera = None
        capture._mss_instance = None

        result = capture._init_backend()
        assert result is False

    def test_capture_region_with_attention(self) -> None:
        capture = ScreenCapture()
        capture.config.region = (0, 0, 1920, 1080)
        capture.set_attention_region(100, 100, 500, 400)

        region = capture._get_capture_region()
        assert region == (100, 100, 500, 400)

    def test_capture_region_without_attention(self) -> None:
        capture = ScreenCapture()
        capture.config.region = (50, 50, 800, 600)

        region = capture._get_capture_region()
        assert region == (50, 50, 800, 600)

    def test_capture_region_default(self) -> None:
        capture = ScreenCapture()
        capture.config.region = None

        region = capture._get_capture_region()
        assert region == (0, 0, 1920, 1080)

    @patch('perception.capture.cv2')
    def test_process_frame_resize(self, mock_cv2) -> None:
        capture = ScreenCapture()
        capture.config.width = 640
        capture.config.height = 480
        capture.config.output_color = "BGR"

        # Mock cv2.resize
        mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.INTER_LINEAR = 1

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        processed = capture._process_frame(frame)

        mock_cv2.resize.assert_called_once()
        assert processed.shape == (480, 640, 3)

    @patch('perception.capture.cv2')
    def test_process_frame_color_convert_bgr_to_rgb(self, mock_cv2) -> None:
        capture = ScreenCapture()
        capture.config.width = 100
        capture.config.height = 100
        capture.config.output_color = "RGB"

        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processed = capture._process_frame(frame)

        mock_cv2.cvtColor.assert_called_once()
        assert processed.shape == (100, 100, 3)

    def test_process_frame_no_resize_needed(self) -> None:
        capture = ScreenCapture()
        capture.config.width = 100
        capture.config.height = 100

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processed = capture._process_frame(frame)

        np.testing.assert_array_equal(processed, frame)


class TestCaptureOnce:
    """Tests for capture_once convenience function."""

    @patch('perception.capture.ScreenCapture')
    def test_capture_once_returns_frame(self, mock_screen_capture_class) -> None:
        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = [None, None, np.zeros((100, 100, 3), dtype=np.uint8)]
        mock_capture.__enter__.return_value = mock_capture
        mock_capture.__exit__.return_value = None
        mock_screen_capture_class.return_value = mock_capture

        frame = capture_once(region=(0, 0, 100, 100))

        assert frame is not None
        assert frame.shape == (100, 100, 3)
        mock_screen_capture_class.assert_called_once()


class TestScreenCaptureAdapter:
    """Tests for ScreenCaptureAdapter."""

    def test_init_with_defaults(self) -> None:
        with patch('perception.capture.ScreenCapture') as mock_cap, \
             patch('perception.state_extractor.StateExtractor') as mock_extractor:
            adapter = ScreenCaptureAdapter()
            assert adapter._capture is not None
            assert adapter._state_extractor is not None

    def test_get_frame_delegates(self) -> None:
        mock_capture = Mock()
        mock_capture.get_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch('perception.state_extractor.StateExtractor'):
            adapter = ScreenCaptureAdapter(capture=mock_capture)
            frame = adapter.get_frame()

        assert frame is not None
        mock_capture.get_frame.assert_called_once()

    def test_extract_delegates(self) -> None:
        mock_extractor = Mock()
        mock_extractor.extract.return_value = "extracted_state"

        mock_capture = Mock()
        mock_capture.get_frame.return_value = None

        with patch('perception.capture.ScreenCapture'):
            adapter = ScreenCaptureAdapter(capture=mock_capture, state_extractor=mock_extractor)

        from core.types import Observation
        obs = Observation(image=np.zeros((100, 100, 3)), timestamp=1.0, metadata={})
        result = adapter.extract(obs)

        assert result == "extracted_state"
        mock_extractor.extract.assert_called_once_with(obs)

    def test_reset_stops_capture(self) -> None:
        mock_capture = Mock()
        mock_capture.stop = Mock()

        with patch('perception.state_extractor.StateExtractor'):
            adapter = ScreenCaptureAdapter(capture=mock_capture)
            adapter.reset()

        mock_capture.stop.assert_called_once()

    def test_properties(self) -> None:
        mock_capture = Mock()
        mock_capture.is_running = True
        mock_capture.config = CaptureConfig(width=640, height=480)

        with patch('perception.state_extractor.StateExtractor'):
            adapter = ScreenCaptureAdapter(capture=mock_capture)

        assert adapter.is_real_time is True
        assert adapter.frame_shape == (480, 640, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])