"""
Test cases for bias_detection.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.bias_detection import BiasDetector, SliceMetrics, BiasReport


class TestSliceMetrics:
    """Test SliceMetrics dataclass"""
    
    def test_slice_metrics_creation(self):
        """Test creating SliceMetrics"""
        metric = SliceMetrics(
            slice_name="test_slice",
            slice_dimension="Test",
            slice_value="value1",
            count=100,
            mae=0.5,
            rmse=0.6,
            mean_predicted=3.5,
            mean_actual=3.6,
            mean_error=0.1,
            std_error=0.2
        )
        
        assert metric.slice_name == "test_slice"
        assert metric.count == 100
        assert metric.mae == 0.5


class TestBiasDetector:
    """Test cases for BiasDetector class"""
    
    @patch('src.bias_detection.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        
        assert detector.project_id == "test-project"
        assert detector.dataset_id == "books"
    
    def test_get_slice_definitions(self):
        """Test getting slice definitions"""
        detector = BiasDetector.__new__(BiasDetector)
        slices = detector.get_slice_definitions()
        
        assert isinstance(slices, list)
        assert len(slices) > 0
        assert all(len(slice_def) == 3 for slice_def in slices)
    
    @patch('src.bias_detection.bigquery.Client')
    def test_compute_slice_metrics(self, mock_client_class):
        """Test computing slice metrics"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High', 'Low'],
            'count': [100, 200],
            'mae': [0.5, 0.6],
            'rmse': [0.7, 0.8],
            'mean_predicted': [3.5, 3.6],
            'mean_actual': [3.4, 3.7],
            'mean_error': [0.1, -0.1],
            'std_error': [0.2, 0.3]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        metrics = detector.compute_slice_metrics(
            "test-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            "popularity_group"
        )
        
        assert len(metrics) == 2
        assert all(isinstance(m, SliceMetrics) for m in metrics)
    
    def test_analyze_disparities(self):
        """Test analyzing disparities"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
            SliceMetrics("slice2", "Popularity", "Low", 200, 0.8, 0.9, 3.6, 3.7, -0.1, 0.3)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert "summary" in result
        assert "detailed_disparities" in result
        assert "high_risk_slices" in result
    
    def test_generate_recommendations(self):
        """Test generating recommendations"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [
                {"severity": "high", "dimension": "Popularity"},
                {"severity": "medium", "dimension": "Era"}
            ],
            "high_risk_slices": [
                {"slice": "test", "mae": 0.9, "mae_deviation_pct": 50}
            ],
            "summary": {
                "Popularity": {"mae_coefficient_of_variation": 0.25}
            }
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    @patch('src.bias_detection.bigquery.Client')
    def test_detect_bias(self, mock_client_class):
        """Test bias detection"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.4],
            'mean_error': [0.1],
            'std_error': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = detector.detect_bias(
            "test-table",
            "test-model",
            "test"
        )
        
        assert isinstance(report, BiasReport)
        assert report.model_name == "test-model"
        assert len(report.slice_metrics) > 0

