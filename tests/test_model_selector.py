"""
Test cases for model_selector.py
"""

import pytest
from unittest.mock import Mock, patch
from src.model_selector import ModelSelector, ModelCandidate, ModelSelectionReport
from src.bias_detection import BiasReport


class TestModelSelector:
    """Test cases for ModelSelector class"""
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_init(self, mock_visualizer, mock_detector):
        """Test initialization"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector(
            performance_weight=0.6,
            fairness_weight=0.4,
            min_fairness_threshold=50.0
        )
        
        assert selector.performance_weight == 0.6
        assert selector.fairness_weight == 0.4
        assert selector.min_fairness_threshold == 50.0
    
    def test_init_invalid_weights(self):
        """Test initialization with invalid weights"""
        with pytest.raises(ValueError):
            ModelSelector(performance_weight=0.6, fairness_weight=0.5)
    
    def test_calculate_fairness_score(self):
        """Test calculating fairness score"""
        selector = ModelSelector.__new__(ModelSelector)
        
        bias_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={
                "detailed_disparities": [
                    {"severity": "high"},
                    {"severity": "medium"}
                ]
            },
            recommendations=[]
        )
        
        score = selector.calculate_fairness_score(bias_report)
        
        # 100 - (1 * 20 + 1 * 10) = 70
        assert score == 70
    
    def test_normalize_performance_score(self):
        """Test normalizing performance score"""
        selector = ModelSelector.__new__(ModelSelector)
        
        # Test with 2 models (percentage-based)
        score = selector.normalize_performance_score(
            mae=0.6,
            rmse=0.7,
            all_maes=[0.5, 0.6],
            all_rmses=[0.6, 0.7]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_calculate_combined_score(self):
        """Test calculating combined score"""
        selector = ModelSelector(
            performance_weight=0.6,
            fairness_weight=0.4
        )
        
        score = selector.calculate_combined_score(
            performance_score=80.0,
            fairness_score=70.0
        )
        
        # 0.6 * 80 + 0.4 * 70 = 48 + 28 = 76
        assert score == 76.0
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_evaluate_model(self, mock_visualizer, mock_detector):
        """Test evaluating model"""
        import pandas as pd
        mock_detector_instance = Mock()
        mock_client = Mock()
        mock_df = pd.DataFrame({'mae': [0.5], 'rmse': [0.6]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_detector_instance.client = mock_client
        mock_detector_instance.detect_bias.return_value = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": []},
            recommendations=[]
        )
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector()
        selector.detector = mock_detector_instance
        
        mae, rmse, bias_report = selector.evaluate_model(
            "test-model",
            "test-table"
        )
        
        assert mae == 0.5
        assert rmse == 0.6
        assert isinstance(bias_report, BiasReport)

