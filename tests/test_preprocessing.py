"""Tests for preprocessing pipeline."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Optional

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


class TestPreprocessConfig:
    """Tests for PreprocessConfig dataclass."""

    def test_default_config(self) -> None:
        config = PreprocessConfig()
        assert config.target_size == (224, 224)
        assert config.normalize is True
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)
        assert config.to_tensor is True
        assert config.to_rgb is True
        assert config.device == DeviceType.AUTO
        assert config.half_precision is False
        assert config.batch_dim is True

    def test_custom_config(self) -> None:
        config = PreprocessConfig(
            target_size=(640, 640),
            normalize=False,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            to_tensor=False,
            to_rgb=False,
            device=DeviceType.CPU,
            half_precision=True,
            batch_dim=False,
        )
        assert config.target_size == (640, 640)
        assert config.normalize is False
        assert config.device == DeviceType.CPU
        assert config.half_precision is True


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types(self) -> None:
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.DIRECTML.value == "directml"
        assert DeviceType.AUTO.value == "auto"


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    @patch('perception.preprocessing.TORCH_AVAILABLE', False)
    @patch('perception.preprocessing.CV2_AVAILABLE', False)
    @patch('perception.preprocessing.DIRECTML_AVAILABLE', False)
    def test_init_cpu_fallback(self) -> None:
        """Test initialization falls back to CPU when no GPU available."""
        config = PreprocessConfig(device=DeviceType.AUTO)
        pipeline = PreprocessingPipeline(config)
        assert pipeline.device == DeviceType.CPU

    def test_init_with_explicit_cpu(self) -> None:
        config = PreprocessConfig(device=DeviceType.CPU)
        pipeline = PreprocessingPipeline(config)
        assert pipeline.device == DeviceType.CPU

    @patch('perception.preprocessing.cv2')
    def test_process_cpu_path(self, mock_cv2) -> None:
        """Test CPU processing path."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1
        mock_cv2.INTER_LINEAR = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process(frame)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 224, 224)  # batch, channels, height, width

    @patch('perception.preprocessing.cv2')
    def test_process_cpu_no_resize_needed(self, mock_cv2) -> None:
        """Test CPU processing when frame already matches target size."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pipeline.process(frame)

        # Should not call resize
        mock_cv2.resize.assert_not_called()
        assert result.shape == (1, 3, 224, 224)

    @patch('perception.preprocessing.cv2')
    def test_process_cpu_no_normalize(self, mock_cv2) -> None:
        """Test CPU processing without normalization."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU, normalize=False)
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pipeline.process(frame)

        # Values should be in 0-255 range (not normalized to 0-1)
        assert result.max() <= 255.0
        assert result.min() >= 0.0

    @patch('perception.preprocessing.cv2')
    def test_process_cpu_no_batch_dim(self, mock_cv2) -> None:
        """Test CPU processing without batch dimension."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU, batch_dim=False)
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pipeline.process(frame)

        assert result.shape == (3, 224, 224)  # No batch dim

    @patch('perception.preprocessing.cv2')
    def test_process_cpu_no_to_tensor(self, mock_cv2) -> None:
        """Test CPU processing without tensor conversion."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1
        mock_cv2.INTER_LINEAR = 1

        config = PreprocessConfig(device=DeviceType.CPU, to_tensor=False)
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process(frame)

        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)  # HWC format

    @patch('perception.preprocessing.cv2')
    def test_process_batch_cpu(self, mock_cv2) -> None:
        """Test batch processing on CPU."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU)
        pipeline = PreprocessingPipeline(config)

        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        result = pipeline.process_batch(frames)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3, 224, 224)

    @patch('perception.preprocessing.cv2')
    def test_process_batch_cpu_empty_raises(self, mock_cv2) -> None:
        """Test batch processing with empty list raises error."""
        config = PreprocessConfig(device=DeviceType.CPU)
        pipeline = PreprocessingPipeline(config)

        with pytest.raises(ValueError, match="Empty frame list"):
            pipeline.process_batch([])


class TestSpecializedPipelines:
    """Tests for factory functions creating specialized pipelines."""

    @patch('perception.preprocessing.cv2')
    def test_create_detection_pipeline(self, mock_cv2) -> None:
        mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        pipeline = create_detection_pipeline()
        assert pipeline.config.target_size == (640, 640)
        assert pipeline.config.mean == (0.0, 0.0, 0.0)
        assert pipeline.config.std == (1.0, 1.0, 1.0)
        assert pipeline.config.half_precision is True

    @patch('perception.preprocessing.cv2')
    def test_create_classification_pipeline(self, mock_cv2) -> None:
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        pipeline = create_classification_pipeline()
        assert pipeline.config.target_size == (224, 224)
        assert pipeline.config.mean == (0.485, 0.456, 0.406)
        assert pipeline.config.std == (0.229, 0.224, 0.225)
        assert pipeline.config.half_precision is False

    @patch('perception.preprocessing.cv2')
    def test_create_ocr_pipeline(self, mock_cv2) -> None:
        mock_cv2.cvtColor.return_value = np.zeros((320, 320, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        pipeline = create_ocr_pipeline()
        assert pipeline.config.target_size == (320, 320)
        assert pipeline.config.mean == (0.5, 0.5, 0.5)
        assert pipeline.config.std == (0.5, 0.5, 0.5)
        assert pipeline.config.half_precision is False


class TestComputeFrameDiff:
    """Tests for compute_frame_diff function."""

    def test_abs_diff(self) -> None:
        frame1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 150, dtype=np.uint8)

        diff = compute_frame_diff(frame1, frame2, method="abs", threshold=0)
        assert diff.shape == (100, 100)  # Grayscale
        assert np.all(diff == 50.0)

    def test_squared_diff(self) -> None:
        frame1 = np.full((100, 100), 100, dtype=np.uint8)
        frame2 = np.full((100, 100), 110, dtype=np.uint8)

        diff = compute_frame_diff(frame1, frame2, method="squared", threshold=0)
        assert np.all(diff == 100.0)  # (110-100)^2 = 100

    def test_relative_diff(self) -> None:
        frame1 = np.full((100, 100), 100, dtype=np.uint8)
        frame2 = np.full((100, 100), 150, dtype=np.uint8)

        diff = compute_frame_diff(frame1, frame2, method="relative", threshold=0)
        # relative = |100-150| / max(100,150) = 50/150 = 1/3
        assert np.allclose(diff, 1.0/3.0)

    def test_threshold_applied(self) -> None:
        frame1 = np.full((100, 100), 100, dtype=np.uint8)
        frame2 = np.full((100, 100), 105, dtype=np.uint8)

        # With threshold=10, diff of 5 should become 0
        diff = compute_frame_diff(frame1, frame2, method="abs", threshold=10)
        assert np.all(diff == 0.0)

        # With threshold=3, diff of 5 should become 1
        diff = compute_frame_diff(frame1, frame2, method="abs", threshold=3)
        assert np.all(diff == 1.0)

    def test_shape_mismatch_raises(self) -> None:
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Frame shapes differ"):
            compute_frame_diff(frame1, frame2)

    def test_invalid_method_raises(self) -> None:
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unknown method"):
            compute_frame_diff(frame1, frame2, method="invalid")


class TestGetMotionRegions:
    """Tests for get_motion_regions function."""

    @patch('perception.preprocessing.cv2')
    def test_get_motion_regions_detects_change(self, mock_cv2) -> None:
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        mock_cv2.findContours.return_value = (
            [np.array([[[30, 20]], [[60, 20]], [[60, 40]], [[30, 40]]], dtype=np.int32)],
            None
        )
        mock_cv2.contourArea.return_value = 600  # Larger than min_area=100
        mock_cv2.boundingRect.return_value = (30, 20, 30, 20)
        mock_cv2.getStructuringElement.return_value = np.ones((5, 5), dtype=np.uint8)
        mock_cv2.morphologyEx.side_effect = lambda *args, **kwargs: args[0]
        mock_cv2.MORPH_ELLIPSE = 1
        mock_cv2.MORPH_OPEN = 1
        mock_cv2.MORPH_CLOSE = 1
        mock_cv2.RETR_EXTERNAL = 1
        mock_cv2.CHAIN_APPROX_SIMPLE = 1

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[20:40, 30:60] = 255

        regions = get_motion_regions(frame1, frame2, min_area=100, threshold=15.0)

        assert len(regions) > 0
        x, y, w, h = regions[0]
        assert x == 30
        assert y == 20
        assert w == 30
        assert h == 20

    @patch('perception.preprocessing.cv2', None)
    def test_get_motion_regions_no_cv2_returns_empty(self) -> None:
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        regions = get_motion_regions(frame1, frame2)
        assert regions == []

    @patch('perception.preprocessing.cv2')
    def test_get_motion_regions_filters_small_areas(self, mock_cv2) -> None:
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        # Small contour (area < min_area)
        mock_cv2.findContours.return_value = (
            [np.array([[[10, 10]], [[12, 10]], [[12, 12]], [[10, 12]]], dtype=np.int32)],
            None
        )
        mock_cv2.boundingRect.return_value = (10, 10, 2, 2)
        mock_cv2.getStructuringElement.return_value = np.ones((5, 5), dtype=np.uint8)
        mock_cv2.morphologyEx.side_effect = lambda *args, **kwargs: args[0]
        mock_cv2.contourArea.return_value = 4  # Small area
        mock_cv2.MORPH_ELLIPSE = 1
        mock_cv2.MORPH_OPEN = 1
        mock_cv2.MORPH_CLOSE = 1
        mock_cv2.RETR_EXTERNAL = 1
        mock_cv2.CHAIN_APPROX_SIMPLE = 1

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        regions = get_motion_regions(frame1, frame2, min_area=100)
        assert len(regions) == 0


class TestPreprocessingPipelineProperties:
    """Tests for pipeline properties."""

    def test_device_property(self) -> None:
        config = PreprocessConfig(device=DeviceType.CPU)
        pipeline = PreprocessingPipeline(config)
        assert pipeline.device == DeviceType.CPU

    @patch('perception.preprocessing.TORCH_AVAILABLE', False)
    def test_torch_device_property_cpu(self) -> None:
        config = PreprocessConfig(device=DeviceType.CPU)
        pipeline = PreprocessingPipeline(config)
        # Should not raise
        device = pipeline.torch_device
        assert str(device) == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])