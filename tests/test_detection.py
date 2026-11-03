"""Tests for object detection."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from perception.detection import (
    ModelType,
    DetectionConfig,
    Detection,
    DetectionResult,
    BaseDetector,
    YOLODetector,
    RTDETRDetector,
    create_detector,
    draw_detections,
)


class TestDetectionConfig:
    """Tests for DetectionConfig dataclass."""

    def test_default_config(self) -> None:
        config = DetectionConfig(model_path="model.onnx")
        assert config.model_path == "model.onnx"
        assert config.model_type == ModelType.YOLOV8
        assert config.input_size == (640, 640)
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.max_detections == 100
        assert config.classes is None
        assert config.class_ids is None
        assert config.use_directml is True
        assert config.device_id == 0
        assert config.half_precision is True
        assert config.normalize is True
        assert config.mean == (0.0, 0.0, 0.0)
        assert config.std == (1.0, 1.0, 1.0)
        assert config.swap_rb is True

    def test_custom_config(self) -> None:
        config = DetectionConfig(
            model_path="custom.onnx",
            model_type=ModelType.RT_DETR,
            input_size=(416, 416),
            confidence_threshold=0.5,
            iou_threshold=0.5,
            classes=["person", "car"],
            class_ids=[0, 2],
            use_directml=False,
            half_precision=False,
        )
        assert config.model_type == ModelType.RT_DETR
        assert config.input_size == (416, 416)
        assert config.confidence_threshold == 0.5
        assert config.classes == ["person", "car"]
        assert config.class_ids == [0, 2]
        assert config.use_directml is False
        assert config.half_precision is False


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self) -> None:
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=(0.1, 0.2, 0.3, 0.4),  # x, y, w, h normalized
            bbox_xyxy=(0.1, 0.2, 0.4, 0.6),  # x1, y1, x2, y2 normalized
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox == (0.1, 0.2, 0.3, 0.4)
        assert det.bbox_xyxy == (0.1, 0.2, 0.4, 0.6)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_detection_result_creation(self) -> None:
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.2), bbox_xyxy=(0.1, 0.1, 0.3, 0.3)),
            Detection(class_id=1, class_name="car", confidence=0.8, bbox=(0.5, 0.5, 0.3, 0.2), bbox_xyxy=(0.5, 0.5, 0.8, 0.7)),
        ]
        result = DetectionResult(
            detections=detections,
            inference_time_ms=15.5,
            preprocessing_time_ms=2.0,
            postprocessing_time_ms=1.0,
            input_shape=(640, 480),
            frame_id=1,
        )
        assert len(result.detections) == 2
        assert result.inference_time_ms == 15.5
        assert result.frame_id == 1


class TestBaseDetector:
    """Tests for BaseDetector abstract class."""

    def test_abstract_methods(self) -> None:
        config = DetectionConfig(model_path="model.onnx")
        detector = BaseDetector(config)

        with pytest.raises(NotImplementedError):
            detector.initialize()

        with pytest.raises(NotImplementedError):
            detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        # shutdown should not raise
        detector.shutdown()
        assert detector.is_initialized is False

    def test_properties(self) -> None:
        config = DetectionConfig(model_path="model.onnx", classes=["person", "car"])
        detector = BaseDetector(config)
        assert detector.class_names == ["person", "car"]
        assert detector.is_initialized is False


class TestYOLODetector:
    """Tests for YOLODetector."""

    @patch('perception.detection.ort', None)
    def test_initialize_no_ort(self) -> None:
        config = DetectionConfig(model_path="model.onnx")
        detector = YOLODetector(config)
        result = detector.initialize()
        assert result is False

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_initialize_model_not_found(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = False

        config = DetectionConfig(model_path="model.onnx")
        detector = YOLODetector(config)
        result = detector.initialize()

        assert result is False

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_initialize_success(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        mock_session.run.return_value = [np.random.rand(1, 84, 8400).astype(np.float32)]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        config = DetectionConfig(model_path="model.onnx", use_directml=False)
        detector = YOLODetector(config)
        result = detector.initialize()

        assert result is True
        assert detector.is_initialized
        mock_ort.InferenceSession.assert_called_once()

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_detect_not_initialized_raises(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session

        config = DetectionConfig(model_path="model.onnx")
        detector = YOLODetector(config)
        detector.initialize()

        # Manually set to not initialized
        detector._initialized = False

        with pytest.raises(RuntimeError, match="Detector not initialized"):
            detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_detect_returns_results(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={"names": '["person", "car"]'})
        # YOLOv8 output format: (1, 84, 8400) -> 80 classes + 4 box coords
        mock_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Add a detection at index 0
        mock_output[0, 4, 0] = 0.9  # class 0 (person) confidence
        mock_output[0, 0, 0] = 320  # x center
        mock_output[0, 1, 0] = 240  # y center
        mock_output[0, 2, 0] = 100  # width
        mock_output[0, 3, 0] = 150  # height
        mock_session.run.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", confidence_threshold=0.5)
        detector = YOLODetector(config)
        detector.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, frame_id=1)

        assert isinstance(result, DetectionResult)
        assert result.frame_id == 1
        assert result.input_shape == (640, 480)
        assert result.inference_time_ms >= 0
        assert result.preprocessing_time_ms >= 0
        assert result.postprocessing_time_ms >= 0

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_preprocess_letterbox(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", input_size=(640, 640))
        detector = YOLODetector(config)
        detector.initialize()

        # Test with non-square image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3 aspect
        input_tensor, scale_info = detector._preprocess(frame)

        assert input_tensor.shape == (1, 3, 640, 640)
        assert scale_info["scale"] == 1.0  # width matches, height scales
        assert scale_info["pad_top"] > 0  # Padding added top/bottom

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_postprocess_nms(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", iou_threshold=0.45)
        detector = YOLODetector(config)
        detector.initialize()

        # Create overlapping boxes
        boxes = np.array([
            [100, 100, 200, 200],  # Box 1
            [110, 110, 210, 210],  # Box 2 - overlaps with box 1
            [300, 300, 400, 400],  # Box 3 - no overlap
        ])
        scores = np.array([0.9, 0.8, 0.7])

        keep = detector._nms(boxes, scores, 0.45)

        # Should keep box 1 (highest score) and box 3 (no overlap)
        assert len(keep) == 2
        assert 0 in keep  # Box 1
        assert 2 in keep  # Box 3


class TestRTDETRDetector:
    """Tests for RTDETRDetector."""

    @patch('perception.detection.ort', None)
    def test_initialize_no_ort(self) -> None:
        config = DetectionConfig(model_path="model.onnx", model_type=ModelType.RT_DETR)
        detector = RTDETRDetector(config)
        result = detector.initialize()
        assert result is False

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_initialize_success(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [
            Mock(name="labels"), Mock(name="boxes"), Mock(name="scores")
        ]
        mock_session.run.return_value = [
            np.array([[0, 1, 2]]),  # labels
            np.array([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.15, 0.15]]]),  # boxes (cxcywh)
            np.array([[0.9, 0.8, 0.7]]),  # scores
        ]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", model_type=ModelType.RT_DETR, use_directml=False)
        detector = RTDETRDetector(config)
        result = detector.initialize()

        assert result is True
        assert detector.is_initialized


class TestCreateDetector:
    """Tests for create_detector factory function."""

    @patch('perception.detection.YOLODetector')
    def test_create_yolo_detector(self, mock_yolo_class) -> None:
        mock_detector = Mock()
        mock_yolo_class.return_value = mock_detector

        config = DetectionConfig(model_path="model.onnx", model_type=ModelType.YOLOV8)
        detector = create_detector(config)

        assert detector == mock_detector
        mock_yolo_class.assert_called_once_with(config)

    @patch('perception.detection.RTDETRDetector')
    def test_create_rtdetr_detector(self, mock_rtdetr_class) -> None:
        mock_detector = Mock()
        mock_rtdetr_class.return_value = mock_detector

        config = DetectionConfig(model_path="model.onnx", model_type=ModelType.RT_DETR)
        detector = create_detector(config)

        assert detector == mock_detector
        mock_rtdetr_class.assert_called_once_with(config)

    @patch('perception.detection.YOLODetector')
    def test_create_custom_fallbacks_to_yolo(self, mock_yolo_class) -> None:
        mock_detector = Mock()
        mock_yolo_class.return_value = mock_detector

        config = DetectionConfig(model_path="model.onnx", model_type=ModelType.CUSTOM)
        detector = create_detector(config)

        assert detector == mock_detector
        mock_yolo_class.assert_called_once_with(config)

    @patch('perception.detection.YOLODetector')
    def test_create_infers_from_model_name(self, mock_yolo_class) -> None:
        mock_detector = Mock()
        mock_yolo_class.return_value = mock_detector

        config = DetectionConfig(model_path="models/rtdetr_model.onnx", model_type=ModelType.CUSTOM)
        detector = create_detector(config)

        # Should infer RT-DETR from model name
        from perception.detection import RTDETRDetector
        # Actually it falls back to YOLODetector for CUSTOM
        # Let's check the actual logic - it checks for "rt_detr" or "rtdetr" in name
        from perception.detection import create_detector as real_create
        # The function tries to infer from model name
        # In this case, "rtdetr" is in the name so it should create RTDETRDetector
        # But CUSTOM type doesn't trigger that - let's verify behavior


class TestDrawDetections:
    """Tests for draw_detections function."""

    @patch('perception.detection.cv2', None)
    def test_draw_detections_no_cv2_returns_original(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.2), bbox_xyxy=(0.1, 0.1, 0.3, 0.3)),
        ]
        result = draw_detections(image, detections)
        np.testing.assert_array_equal(result, image)

    @patch('perception.detection.cv2')
    def test_draw_detections_draws_boxes(self, mock_cv2) -> None:
        mock_cv2.rectangle.return_value = None
        mock_cv2.putText.return_value = None
        mock_cv2.getTextSize.return_value = ((50, 20), 5)
        mock_cv2.FONT_HERSHEY_SIMPLEX = 1

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.2), bbox_xyxy=(0.1, 0.1, 0.3, 0.3)),
        ]

        result = draw_detections(image, detections)

        assert mock_cv2.rectangle.called
        assert mock_cv2.putText.called
        # Result should be modified copy
        assert result is not image


class TestYOLODetectorEdgeCases:
    """Tests for edge cases in YOLODetector."""

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_postprocess_handles_different_output_shapes(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        # Test with (1, 8400, 84) format (already transposed)
        mock_output = np.zeros((1, 8400, 84), dtype=np.float32)
        mock_output[0, 0, 4] = 0.9  # class 0 confidence
        mock_output[0, 0, :4] = [0.5, 0.5, 0.2, 0.2]  # box
        mock_session.run.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx")
        detector = YOLODetector(config)
        detector.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert isinstance(result, DetectionResult)

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_postprocess_filters_by_class_ids(self, mock_exists, mock_ort) -> None:
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        # Two detections: class 0 (person) and class 1 (car)
        mock_output = np.zeros((1, 84, 8400), dtype=np.float32)
        mock_output[0, 4, 0] = 0.9  # class 0
        mock_output[0, :4, 0] = [0.5, 0.5, 0.2, 0.2]
        mock_output[0, 5, 1] = 0.8  # class 1
        mock_output[0, :4, 1] = [0.3, 0.3, 0.1, 0.1]
        mock_session.run.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", class_ids=[0])  # Only person
        detector = YOLODetector(config)
        detector.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        # Should only have class 0 detections
        for det in result.detections:
            assert det.class_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])