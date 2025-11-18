"""
Comprehensive test cases for model_selector.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.model_selector import ModelSelector, ModelCandidate, ModelSelectionReport
from src.bias_detection import BiasReport, SliceMetrics


class TestModelSelector:
    """Comprehensive test cases for ModelSelector class"""
    
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
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_init_default_values(self, mock_visualizer, mock_detector):
        """Test initialization with default values"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector()
        
        assert selector.performance_weight == 0.6
        assert selector.fairness_weight == 0.4
        assert selector.min_fairness_threshold == 50.0
    
    def test_init_invalid_weights(self):
        """Test initialization with invalid weights"""
        with pytest.raises(ValueError):
            ModelSelector(performance_weight=0.6, fairness_weight=0.5)
    
    def test_init_weights_not_sum_to_one(self):
        """Test initialization when weights don't sum to 1.0"""
        with pytest.raises(ValueError):
            ModelSelector(performance_weight=0.7, fairness_weight=0.4)
    
    def test_init_zero_weights(self):
        """Test initialization with zero weights"""
        with pytest.raises(ValueError):
            ModelSelector(performance_weight=0.0, fairness_weight=0.0)
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_fairness_score_no_disparities(self, mock_visualizer, mock_detector):
        """Test calculating fairness score with no disparities"""
        selector = ModelSelector.__new__(ModelSelector)
        
        bias_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": []},
            recommendations=[]
        )
        
        score = selector.calculate_fairness_score(bias_report)
        
        assert score == 100
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_fairness_score_high_severity(self, mock_visualizer, mock_detector):
        """Test calculating fairness score with high severity disparities"""
        selector = ModelSelector.__new__(ModelSelector)
        
        bias_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={
                "detailed_disparities": [
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"}
                ]
            },
            recommendations=[]
        )
        
        score = selector.calculate_fairness_score(bias_report)
        
        # 100 - (3 * 20) = 40
        assert score == 40
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_fairness_score_mixed_severity(self, mock_visualizer, mock_detector):
        """Test calculating fairness score with mixed severity"""
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
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_fairness_score_negative_result(self, mock_visualizer, mock_detector):
        """Test calculating fairness score that would be negative (should be capped at 0)"""
        selector = ModelSelector.__new__(ModelSelector)
        
        bias_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={
                "detailed_disparities": [
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"}
                ]
            },
            recommendations=[]
        )
        
        score = selector.calculate_fairness_score(bias_report)
        
        # Should be capped at 0
        assert score == 0
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_normalize_performance_score_two_models(self, mock_visualizer, mock_detector):
        """Test normalizing performance score with 2 models (percentage-based)"""
        selector = ModelSelector.__new__(ModelSelector)
        
        score = selector.normalize_performance_score(
            mae=0.6,
            rmse=0.7,
            all_maes=[0.5, 0.6],
            all_rmses=[0.6, 0.7]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_normalize_performance_score_three_models(self, mock_visualizer, mock_detector):
        """Test normalizing performance score with 3+ models (min-max)"""
        selector = ModelSelector.__new__(ModelSelector)
        
        score = selector.normalize_performance_score(
            mae=0.6,
            rmse=0.7,
            all_maes=[0.4, 0.6, 0.8],
            all_rmses=[0.5, 0.7, 0.9]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_normalize_performance_score_same_values(self, mock_visualizer, mock_detector):
        """Test normalizing performance score when all values are the same"""
        selector = ModelSelector.__new__(ModelSelector)
        
        score = selector.normalize_performance_score(
            mae=0.5,
            rmse=0.6,
            all_maes=[0.5, 0.5],
            all_rmses=[0.6, 0.6]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_normalize_performance_score_zero_range(self, mock_visualizer, mock_detector):
        """Test normalizing performance score with zero range"""
        selector = ModelSelector.__new__(ModelSelector)
        
        score = selector.normalize_performance_score(
            mae=0.5,
            rmse=0.6,
            all_maes=[0.5, 0.5, 0.5],
            all_rmses=[0.6, 0.6, 0.6]
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_combined_score(self, mock_visualizer, mock_detector):
        """Test calculating combined score"""
        selector = ModelSelector(
            performance_weight=0.6,
            fairness_weight=0.4
        )
        
        score = selector.calculate_combined_score(
            performance_score=80.0,
            fairness_score=70.0
        )
        
        assert score == 76.0
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_combined_score_equal_weights(self, mock_visualizer, mock_detector):
        """Test calculating combined score with equal weights"""
        selector = ModelSelector(
            performance_weight=0.5,
            fairness_weight=0.5
        )
        
        score = selector.calculate_combined_score(
            performance_score=80.0,
            fairness_score=70.0
        )
        
        assert score == 75.0
    
    @patch("src.model_selector.BiasDetector")
    @patch("src.model_selector.BiasVisualizer")
    def test_calculate_combined_score_performance_focused(self, mock_visualizer, mock_detector):
        """Test calculating combined score with performance-focused weights"""
        selector = ModelSelector(
            performance_weight=0.8,
            fairness_weight=0.2
        )
        
        score = selector.calculate_combined_score(
            performance_score=80.0,
            fairness_score=70.0
        )
        
        assert score == 78.0
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_evaluate_model_success(self, mock_visualizer, mock_detector):
        """Test evaluating model successfully"""
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
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_evaluate_model_exception(self, mock_visualizer, mock_detector):
        """Test exception handling in evaluate_model"""
        mock_detector_instance = Mock()
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_detector_instance.client = mock_client
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector()
        selector.detector = mock_detector_instance
        
        with pytest.raises(Exception):
            selector.evaluate_model("test-model", "test-table")
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_compare_models_success(self, mock_visualizer, mock_detector):
        """Test comparing models successfully"""
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
        
        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        
        selector = ModelSelector()
        selector.detector = mock_detector_instance
        selector.visualizer = mock_visualizer_instance
        
        model_candidates = [
            {'model_name': 'model1', 'predictions_table': 'table1'},
            {'model_name': 'model2', 'predictions_table': 'table2'}
        ]
        
        with patch.object(selector, '_generate_selection_report') as mock_report:
            with patch.object(selector, '_print_selection_summary'):
                with patch.object(selector, '_save_selection_report'):
                    with patch.object(selector, '_generate_comparison_visualizations'):
                        mock_report.return_value = Mock(
                            selected_model=Mock(model_name='model1')
                        )
                        report = selector.compare_models(model_candidates)
                        assert report is not None
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_compare_models_no_candidates(self, mock_visualizer, mock_detector):
        """Test comparing models with no candidates"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector()
        selector.detector = mock_detector_instance
        
        with pytest.raises(ValueError):
            selector.compare_models([])
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_compare_models_all_fail(self, mock_visualizer, mock_detector):
        """Test comparing models when all evaluations fail"""
        mock_detector_instance = Mock()
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_detector_instance.client = mock_client
        mock_detector.return_value = mock_detector_instance
        
        selector = ModelSelector()
        selector.detector = mock_detector_instance
        
        model_candidates = [
            {'model_name': 'model1', 'predictions_table': 'table1'},
            {'model_name': 'model2', 'predictions_table': 'table2'}
        ]
        
        with pytest.raises(ValueError):
            selector.compare_models(model_candidates)
    
    @patch('src.model_selector.BiasDetector')
    @patch('src.model_selector.BiasVisualizer')
    def test_compare_models_fairness_threshold(self, mock_visualizer, mock_detector):
        """Test comparing models with fairness threshold"""
        mock_detector_instance = Mock()
        mock_client = Mock()
        mock_df = pd.DataFrame({'mae': [0.5], 'rmse': [0.6]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_detector_instance.client = mock_client
        
        # Model 1: high fairness
        bias_report1 = BiasReport(
            timestamp="2025-01-01",
            model_name="model1",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": []},
            recommendations=[]
        )
        
        # Model 2: low fairness
        bias_report2 = BiasReport(
            timestamp="2025-01-01",
            model_name="model2",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={
                "detailed_disparities": [
                    {"severity": "high"},
                    {"severity": "high"},
                    {"severity": "high"}
                ]
            },
            recommendations=[]
        )
        
        mock_detector_instance.detect_bias.side_effect = [bias_report1, bias_report2]
        mock_detector.return_value = mock_detector_instance
        
        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        
        selector = ModelSelector(
            min_fairness_threshold=60.0
        )
        selector.detector = mock_detector_instance
        selector.visualizer = mock_visualizer_instance
        
        model_candidates = [
            {'model_name': 'model1', 'predictions_table': 'table1'},
            {'model_name': 'model2', 'predictions_table': 'table2'}
        ]
        
        with patch.object(selector, '_generate_selection_report') as mock_report:
            with patch.object(selector, '_print_selection_summary'):
                with patch.object(selector, '_save_selection_report'):
                    with patch.object(selector, '_generate_comparison_visualizations'):
                        mock_report.return_value = Mock(
                            selected_model=Mock(model_name='model1')
                        )
                        report = selector.compare_models(model_candidates)
                        assert report is not None
