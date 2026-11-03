"""Tests for OCR and HUD parsing."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Tuple

from perception.ocr import (
    OCREngine,
    OCRConfig,
    OCRResult,
    HUDData,
    OCREngineBase,
    PaddleOCREngine,
    TesseractEngine,
    HUDParser,
    OCRProcessor,
    create_ocr_processor,
)


class TestOCRConfig:
    """Tests for OCRConfig dataclass."""

    def test_default_config(self) -> None:
        config = OCRConfig()
        assert config.engine == OCREngine.AUTO
        assert config.languages == ["en"]
        assert config.use_gpu is True
        assert config.drop_score == 0.5
        assert config.tesseract_config == "--oem 3 --psm 6"
        assert config.preprocess is True
        assert config.upscale_factor == 2.0
        assert config.ignore_regions == []

    def test_custom_config(self) -> None:
        config = OCRConfig(
            engine=OCREngine.PADDLEOCR,
            languages=["en", "zh"],
            use_gpu=False,
            drop_score=0.7,
            preprocess=False,
            upscale_factor=1.5,
            ignore_regions=[(0, 0, 100, 100)],
        )
        assert config.engine == OCREngine.PADDLEOCR
        assert config.languages == ["en", "zh"]
        assert config.use_gpu is False
        assert config.drop_score == 0.7
        assert config.preprocess is False
        assert config.upscale_factor == 1.5
        assert config.ignore_regions == [(0, 0, 100, 100)]


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_ocr_result_creation(self) -> None:
        result = OCRResult(
            text="HP: 100/100",
            confidence=0.95,
            bbox=(10, 20, 100, 30),
            engine="paddleocr",
            raw_result={"test": "data"},
        )
        assert result.text == "HP: 100/100"
        assert result.confidence == 0.95
        assert result.bbox == (10, 20, 100, 30)
        assert result.engine == "paddleocr"
        assert result.raw_result == {"test": "data"}


class TestHUDData:
    """Tests for HUDData dataclass."""

    def test_default_hud_data(self) -> None:
        hud = HUDData()
        assert hud.health is None
        assert hud.max_health is None
        assert hud.mana is None
        assert hud.gold is None
        assert hud.custom == {}
        assert hud.raw_results == []

    def test_hud_data_with_values(self) -> None:
        hud = HUDData(
            health=100.0,
            max_health=100.0,
            gold=1234,
            custom={"boss_hp": 5000},
        )
        assert hud.health == 100.0
        assert hud.max_health == 100.0
        assert hud.gold == 1234
        assert hud.custom == {"boss_hp": 5000}


class TestOCREngineBase:
    """Tests for OCREngineBase abstract class."""

    def test_abstract_methods(self) -> None:
        config = OCRConfig()
        engine = OCREngineBase(config)

        with pytest.raises(NotImplementedError):
            engine.initialize()

        with pytest.raises(NotImplementedError):
            engine.recognize(np.zeros((100, 100, 3), dtype=np.uint8))

        # shutdown should not raise
        engine.shutdown()


class TestPaddleOCREngine:
    """Tests for PaddleOCREngine."""

    @patch('perception.ocr.PaddleOCR', None)
    def test_initialize_no_paddleocr(self) -> None:
        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        engine = PaddleOCREngine(config)
        result = engine.initialize()
        assert result is False

    @patch('perception.ocr.PaddleOCR')
    @patch('perception.ocr.cv2')
    def test_initialize_success(self, mock_cv2, mock_paddleocr_class) -> None:
        mock_ocr = Mock()
        mock_paddleocr_class.return_value = mock_ocr

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        engine = PaddleOCREngine(config)
        result = engine.initialize()

        assert result is True
        mock_paddleocr_class.assert_called_once()

    @patch('perception.ocr.PaddleOCR')
    @patch('perception.ocr.cv2')
    def test_recognize_returns_results(self, mock_cv2, mock_paddleocr_class) -> None:
        mock_ocr = Mock()
        # PaddleOCR returns list of list of [bbox, (text, confidence)]
        mock_ocr.ocr.return_value = [[
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("HP: 100/100", 0.95)],
            [[[200, 30], [300, 30], [300, 60], [200, 60]], ("Gold: 1,234", 0.90)],
        ]]
        mock_paddleocr_class.return_value = mock_ocr

        mock_cv2.boundingRect.return_value = (10, 20, 100, 30)
        mock_cv2.COLOR_BGR2GRAY = 1
        mock_cv2.fastNlMeansDenoising.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.createCLAHE.return_value = Mock(apply=Mock(return_value=np.zeros((100, 100), dtype=np.uint8)))
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        mock_cv2.COLOR_GRAY2BGR = 1
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_cv2.INTER_CUBIC = 1

        config = OCRConfig(engine=OCREngine.PADDLEOCR, preprocess=True, upscale_factor=2.0)
        engine = PaddleOCREngine(config)
        engine.initialize()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = engine.recognize(frame)

        assert len(results) == 2
        assert results[0].text == "HP: 100/100"
        assert results[0].confidence == 0.95
        assert results[0].engine == "paddleocr"
        assert results[1].text == "Gold: 1,234"

    @patch('perception.ocr.PaddleOCR')
    @patch('perception.ocr.cv2')
    def test_recognize_filters_low_confidence(self, mock_cv2, mock_paddleocr_class) -> None:
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [[
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("HP: 100/100", 0.30)],  # Below drop_score
            [[[200, 30], [300, 30], [300, 60], [200, 60]], ("Gold: 1,234", 0.90)],
        ]]
        mock_paddleocr_class.return_value = mock_ocr

        mock_cv2.boundingRect.return_value = (10, 20, 100, 30)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_cv2.INTER_CUBIC = 1
        mock_cv2.COLOR_BGR2GRAY = 1
        mock_cv2.fastNlMeansDenoising.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.createCLAHE.return_value = Mock(apply=Mock(return_value=np.zeros((100, 100), dtype=np.uint8)))
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        mock_cv2.COLOR_GRAY2BGR = 1

        config = OCRConfig(engine=OCREngine.PADDLEOCR, drop_score=0.5)
        engine = PaddleOCREngine(config)
        engine.initialize()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = engine.recognize(frame)

        # Only high confidence result should be returned
        assert len(results) == 1
        assert results[0].text == "Gold: 1,234"


class TestTesseractEngine:
    """Tests for TesseractEngine."""

    @patch('perception.ocr.pytesseract', None)
    def test_initialize_no_tesseract(self) -> None:
        config = OCRConfig(engine=OCREngine.TESSERACT)
        engine = TesseractEngine(config)
        result = engine.initialize()
        assert result is False

    @patch('perception.ocr.pytesseract')
    def test_initialize_success(self, mock_pytesseract) -> None:
        mock_pytesseract.get_tesseract_version.return_value = "5.0.0"

        config = OCRConfig(engine=OCREngine.TESSERACT)
        engine = TesseractEngine(config)
        result = engine.initialize()

        assert result is True

    @patch('perception.ocr.pytesseract')
    @patch('perception.ocr.cv2')
    def test_recognize_returns_results(self, mock_cv2, mock_pytesseract) -> None:
        mock_pytesseract.get_tesseract_version.return_value = "5.0.0"
        mock_pytesseract.image_to_data.return_value = {
            "text": ["HP:", "100/100", "Gold:", "1,234"],
            "conf": ["95", "95", "90", "90"],
            "left": [10, 40, 200, 250],
            "top": [20, 20, 30, 30],
            "width": [30, 60, 40, 80],
            "height": [30, 30, 30, 30],
        }
        mock_pytesseract.Output.DICT = "dict"

        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((200, 200), dtype=np.uint8)
        mock_cv2.INTER_CUBIC = 1
        mock_cv2.COLOR_BGR2GRAY = 1
        mock_cv2.fastNlMeansDenoising.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))

        config = OCRConfig(engine=OCREngine.TESSERACT)
        engine = TesseractEngine(config)
        engine.initialize()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = engine.recognize(frame)

        assert len(results) == 4
        assert results[0].text == "HP:"
        assert results[0].confidence == 0.95
        assert results[0].engine == "tesseract"


class TestHUDParser:
    """Tests for HUDParser."""

    def test_parse_health_patterns(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="HP: 100/100", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="Health 50/100", confidence=0.9, bbox=(10, 50, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.health == 100.0
        assert hud.max_health == 100.0

    def test_parse_mana_patterns(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="MP: 50/100", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="Mana 75/100", confidence=0.9, bbox=(10, 50, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.mana == 50.0
        assert hud.max_mana == 100.0

    def test_parse_level_pattern(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Level: 25", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="Lv 10", confidence=0.9, bbox=(10, 50, 50, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.level == 25  # First match wins

    def test_parse_gold_pattern(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Gold: 1,234", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="5,000 Gold", confidence=0.9, bbox=(10, 50, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.gold == 1234  # First match wins

    def test_parse_score_pattern(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Score: 10000", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.score == 10000

    def test_parse_ammo_pattern(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Ammo: 30/90", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.ammo == 30
        assert hud.max_ammo == 90

    def test_parse_kills_deaths(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Kills: 10", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="Deaths: 3", confidence=0.9, bbox=(10, 50, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.kills == 10
        assert hud.deaths == 3

    def test_parse_time_pattern(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Time: 10:30", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.time_remaining == "10:30"

    def test_parse_custom_key_value(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="BossHP: 5000", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
            OCRResult(text="CustomValue = 42", confidence=0.9, bbox=(10, 50, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert "bosshp" in hud.custom
        assert "customvalue" in hud.custom

    def test_parse_stamina(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="Stamina 80/100", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.stamina == 80.0
        assert hud.max_stamina == 100.0

    def test_parse_experience(self) -> None:
        parser = HUDParser()
        results = [
            OCRResult(text="EXP: 1500/2000", confidence=0.9, bbox=(10, 10, 100, 30), engine="test"),
        ]
        hud = parser.parse(results)
        assert hud.experience == 1500.0


class TestOCRProcessor:
    """Tests for OCRProcessor."""

    @patch('perception.ocr.PADDLEOCR_AVAILABLE', False)
    @patch('perception.ocr.TESSERACT_AVAILABLE', False)
    def test_initialize_no_engines(self) -> None:
        config = OCRConfig()
        processor = OCRProcessor(config)
        result = processor.initialize()
        assert result is False

    @patch('perception.ocr.PaddleOCREngine')
    def test_initialize_with_paddleocr(self, mock_paddle_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_paddle_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        result = processor.initialize()

        assert result is True
        mock_paddle_engine_class.assert_called_once_with(config)
        mock_engine.initialize.assert_called_once()

    @patch('perception.ocr.TesseractEngine')
    def test_initialize_with_tesseract(self, mock_tesseract_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_tesseract_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)
        result = processor.initialize()

        assert result is True

    @patch('perception.ocr.PaddleOCREngine')
    def test_recognize_delegates_to_engine(self, mock_paddle_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_engine.recognize.return_value = [
            OCRResult(text="HP: 100/100", confidence=0.95, bbox=(10, 10, 100, 30), engine="paddleocr"),
        ]
        mock_paddle_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = processor.recognize(frame)

        assert len(results) == 1
        assert results[0].text == "HP: 100/100"
        mock_engine.recognize.assert_called_once_with(frame)

    @patch('perception.ocr.PaddleOCREngine')
    def test_extract_hud(self, mock_paddle_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_engine.recognize.return_value = [
            OCRResult(text="HP: 100/100", confidence=0.95, bbox=(10, 10, 100, 30), engine="paddleocr"),
            OCRResult(text="Gold: 1,234", confidence=0.90, bbox=(200, 10, 100, 30), engine="paddleocr"),
        ]
        mock_paddle_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hud = processor.extract_hud(frame)

        assert hud.health == 100.0
        assert hud.max_health == 100.0
        assert hud.gold == 1234
        assert len(hud.raw_results) == 2

    @patch('perception.ocr.PaddleOCREngine')
    def test_extract_text_regions(self, mock_paddle_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_engine.recognize.side_effect = [
            [OCRResult(text="HP: 100", confidence=0.9, bbox=(0, 0, 50, 20), engine="paddleocr")],
            [OCRResult(text="Gold: 500", confidence=0.9, bbox=(0, 0, 50, 20), engine="paddleocr")],
        ]
        mock_paddle_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        regions = [(10, 10, 100, 30), (10, 50, 100, 30)]
        results = processor.extract_text_regions(frame, regions)

        assert len(results) == 2
        assert results[(10, 10, 100, 30)] == "HP: 100"
        assert results[(10, 50, 100, 30)] == "Gold: 500"

    @patch('perception.ocr.PaddleOCREngine')
    def test_shutdown(self, mock_paddle_engine_class) -> None:
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_paddle_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()
        processor.shutdown()

        mock_engine.shutdown.assert_called_once()


class TestCreateOCRProcessor:
    """Tests for create_ocr_processor factory function."""

    def test_create_with_defaults(self) -> None:
        with patch('perception.ocr.PaddleOCREngine'), \
             patch('perception.ocr.TesseractEngine'):
            processor = create_ocr_processor()
            assert processor is not None
            assert processor.config.engine == OCREngine.AUTO

    def test_create_with_custom_params(self) -> None:
        with patch('perception.ocr.PaddleOCREngine'), \
             patch('perception.ocr.TesseractEngine'):
            processor = create_ocr_processor(
                engine=OCREngine.TESSERACT,
                languages=["en", "fr"],
                use_gpu=False,
            )
            assert processor.config.engine == OCREngine.TESSERACT
            assert processor.config.languages == ["en", "fr"]
            assert processor.config.use_gpu is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])