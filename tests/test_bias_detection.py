"""
Comprehensive test cases for bias_detection.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from src.bias_detection import BiasDetector, SliceMetrics, BiasReport


class TestSliceMetrics:
    """Test SliceMetrics dataclass with all scenarios"""
    
    def test_slice_metrics_creation(self):
        """Test creating SliceMetrics with valid data"""
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
        assert metric.rmse == 0.6
        assert metric.mean_predicted == 3.5
        assert metric.mean_actual == 3.6
        assert metric.mean_error == 0.1
        assert metric.std_error == 0.2
    
    def test_slice_metrics_with_zero_values(self):
        """Test SliceMetrics with zero values"""
        metric = SliceMetrics(
            slice_name="zero_slice",
            slice_dimension="Test",
            slice_value="zero",
            count=0,
            mae=0.0,
            rmse=0.0,
            mean_predicted=0.0,
            mean_actual=0.0,
            mean_error=0.0,
            std_error=0.0
        )
        assert metric.count == 0
        assert metric.mae == 0.0
    
    def test_slice_metrics_with_negative_error(self):
        """Test SliceMetrics with negative error values"""
        metric = SliceMetrics(
            slice_name="negative_slice",
            slice_dimension="Test",
            slice_value="negative",
            count=50,
            mae=0.3,
            rmse=0.4,
            mean_predicted=3.0,
            mean_actual=3.5,
            mean_error=-0.5,
            std_error=0.1
        )
        assert metric.mean_error == -0.5
    
    def test_slice_metrics_with_large_values(self):
        """Test SliceMetrics with large count values"""
        metric = SliceMetrics(
            slice_name="large_slice",
            slice_dimension="Test",
            slice_value="large",
            count=1000000,
            mae=1.5,
            rmse=2.0,
            mean_predicted=4.5,
            mean_actual=4.0,
            mean_error=0.5,
            std_error=1.0
        )
        assert metric.count == 1000000


class TestBiasDetector:
    """Comprehensive test cases for BiasDetector class"""
    
    @patch('src.bias_detection.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization with AIRFLOW_HOME set"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        
        assert detector.project_id == "test-project"
        assert detector.dataset_id == "books"
    
    @patch('src.bias_detection.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            detector = BiasDetector()
            assert detector.project_id == "test-project"
    
    @patch('src.bias_detection.bigquery.Client')
    def test_init_with_custom_project_id(self, mock_client_class):
        """Test initialization with custom project_id"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector(project_id="custom-project")
        assert detector.project_id == "custom-project"
    
    def test_get_slice_definitions(self):
        """Test getting slice definitions"""
        detector = BiasDetector.__new__(BiasDetector)
        slices = detector.get_slice_definitions()
        
        assert isinstance(slices, list)
        assert len(slices) > 0
        assert all(len(slice_def) == 3 for slice_def in slices)
        
        # Check specific dimensions exist
        dimension_names = [s[0] for s in slices]
        assert "Popularity" in dimension_names
        assert "Book Length" in dimension_names
        assert "Book Era" in dimension_names
        assert "Author Gender" in dimension_names
    
    @patch('src.bias_detection.bigquery.Client')
    def test_compute_slice_metrics_success(self, mock_client_class):
        """Test computing slice metrics successfully"""
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
        assert metrics[0].slice_value == "High"
        assert metrics[1].slice_value == "Low"
    
    @patch('src.bias_detection.bigquery.Client')
    def test_compute_slice_metrics_empty_result(self, mock_client_class):
        """Test computing slice metrics with empty result"""
        mock_client = Mock()
        mock_df = pd.DataFrame()
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
        
        assert len(metrics) == 0
    
    @patch('src.bias_detection.bigquery.Client')
    def test_compute_slice_metrics_with_nan_std_error(self, mock_client_class):
        """Test computing slice metrics with NaN std_error"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mae': [0.5],
            'rmse': [0.6],
            'mean_predicted': [3.5],
            'mean_actual': [3.4],
            'mean_error': [0.1],
            'std_error': [np.nan]
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
        
        assert len(metrics) == 1
        assert metrics[0].std_error == 0.0
    
    @patch('src.bias_detection.bigquery.Client')
    def test_compute_slice_metrics_exception_handling(self, mock_client_class):
        """Test exception handling in compute_slice_metrics"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        metrics = detector.compute_slice_metrics(
            "test-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            "popularity_group"
        )
        
        assert len(metrics) == 0
    
    def test_analyze_disparities_single_slice(self):
        """Test analyzing disparities with single slice (should skip)"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert "summary" in result
        assert len(result["summary"]) == 0  # Should skip single-slice dimensions
    
    def test_analyze_disparities_multiple_slices(self):
        """Test analyzing disparities with multiple slices"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
            SliceMetrics("slice2", "Popularity", "Low", 200, 0.8, 0.9, 3.6, 3.7, -0.1, 0.3)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert "summary" in result
        assert "detailed_disparities" in result
        assert "high_risk_slices" in result
        assert "Popularity" in result["summary"]
    
    def test_analyze_disparities_high_cv(self):
        """Test analyzing disparities with high coefficient of variation"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.3, 0.4, 3.5, 3.4, 0.1, 0.2),
            SliceMetrics("slice2", "Popularity", "Low", 200, 0.9, 1.0, 3.6, 3.7, -0.1, 0.3)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert len(result["detailed_disparities"]) > 0
        assert any(d["dimension"] == "Popularity" for d in result["detailed_disparities"])
    
    def test_analyze_disparities_high_risk_slices(self):
        """Test identifying high-risk slices"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
            SliceMetrics("slice2", "Popularity", "Low", 200, 1.0, 1.1, 3.6, 3.7, -0.1, 0.3)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert len(result["high_risk_slices"]) > 0
        # Check for slice2 which has higher MAE (1.0) compared to slice1 (0.5)
        # slice2 should be flagged as high risk since 1.0 > 0.75 * 1.2 (avg_mae * 1.2)
        assert any(s["slice"] == "slice2" for s in result["high_risk_slices"])
    
    def test_analyze_disparities_zero_mean_mae(self):
        """Test analyzing disparities when mean MAE is zero"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.0, 0.0, 3.5, 3.4, 0.0, 0.0),
            SliceMetrics("slice2", "Popularity", "Low", 200, 0.0, 0.0, 3.6, 3.7, 0.0, 0.0)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert "summary" in result
        assert result["summary"]["Popularity"]["mae_coefficient_of_variation"] == 0.0
    
    def test_generate_recommendations_no_bias(self):
        """Test generating recommendations when no bias detected"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [],
            "high_risk_slices": [],
            "summary": {}
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert "No significant bias detected" in recommendations[0]
    
    def test_generate_recommendations_high_severity(self):
        """Test generating recommendations with high severity disparities"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [
                {"severity": "high", "dimension": "Popularity"},
                {"severity": "high", "dimension": "Era"}
            ],
            "high_risk_slices": [],
            "summary": {}
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert any("HIGH PRIORITY" in rec for rec in recommendations)
    
    def test_generate_recommendations_medium_severity(self):
        """Test generating recommendations with medium severity disparities"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [
                {"severity": "medium", "dimension": "Era"}
            ],
            "high_risk_slices": [],
            "summary": {}
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert any("MEDIUM PRIORITY" in rec for rec in recommendations)
    
    def test_generate_recommendations_high_risk_slices(self):
        """Test generating recommendations with high-risk slices"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [],
            "high_risk_slices": [
                {"slice": "test_slice", "mae": 0.9, "mae_deviation_pct": 50},
                {"slice": "test_slice2", "mae": 1.0, "mae_deviation_pct": 60}
            ],
            "summary": {}
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert any("Target these high-error slices" in rec for rec in recommendations)
    
    def test_generate_recommendations_high_cv(self):
        """Test generating recommendations with high coefficient of variation"""
        detector = BiasDetector.__new__(BiasDetector)
        
        disparity_analysis = {
            "detailed_disparities": [],
            "high_risk_slices": [],
            "summary": {
                "Popularity": {"mae_coefficient_of_variation": 0.25, "num_slices": 3}
            }
        }
        
        recommendations = detector.generate_recommendations(disparity_analysis)
        
        assert isinstance(recommendations, list)
        assert any("Popularity" in rec for rec in recommendations)
    
    @patch('src.bias_detection.bigquery.Client')
    def test_detect_bias_success(self, mock_client_class):
        """Test successful bias detection"""
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
        assert report.dataset == "test"
        assert len(report.slice_metrics) > 0
        assert isinstance(report.timestamp, str)
    
    @patch('src.bias_detection.bigquery.Client')
    def test_detect_bias_empty_results(self, mock_client_class):
        """Test bias detection with empty query results"""
        mock_client = Mock()
        mock_df = pd.DataFrame()
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
        assert len(report.slice_metrics) == 0
    
    @patch('src.bias_detection.bigquery.Client')
    def test_detect_bias_with_exception(self, mock_client_class):
        """Test bias detection with query exception"""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = detector.detect_bias(
            "test-table",
            "test-model",
            "test"
        )
        
        assert isinstance(report, BiasReport)
        assert len(report.slice_metrics) == 0
    
    @patch('src.bias_detection.bigquery.Client')
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_report(self, mock_makedirs, mock_open, mock_client_class):
        """Test saving bias report to JSON"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={},
            recommendations=[]
        )
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        detector.save_report(report, "/path/to/report.json")
        
        mock_makedirs.assert_called_once()
        mock_file.write.assert_called()
        # Collect all write calls and join them to reconstruct the JSON
        written_content = ''.join([call[0][0] for call in mock_file.write.call_args_list])
        written_data = json.loads(written_content)
        assert written_data["model_name"] == "test-model"
    
    @patch('src.bias_detection.bigquery.Client')
    def test_create_bias_metrics_table_empty_metrics(self, mock_client_class):
        """Test creating bias metrics table with empty metrics"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={},
            recommendations=[]
        )
        
        detector.create_bias_metrics_table(report, "test-table")
        # Should not raise exception, just skip writing
    
    @patch('src.bias_detection.bigquery.Client')
    def test_create_bias_metrics_table_with_metrics(self, mock_client_class):
        """Test creating bias metrics table with metrics"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.load_table_from_dataframe.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[
                SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2)
            ],
            disparity_analysis={},
            recommendations=[]
        )
        
        detector.create_bias_metrics_table(report, "test-table")
        mock_client.load_table_from_dataframe.assert_called_once()
    
    @patch('src.bias_detection.bigquery.Client')
    def test_create_bias_metrics_table_exception(self, mock_client_class):
        """Test creating bias metrics table with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.load_table_from_dataframe.side_effect = Exception("Load failed")
        mock_client_class.return_value = mock_client
        
        detector = BiasDetector()
        detector.client = mock_client
        
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[
                SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2)
            ],
            disparity_analysis={},
            recommendations=[]
        )
        
        # Should not raise exception, just print error
        detector.create_bias_metrics_table(report, "test-table")
    
    def test_analyze_disparities_multiple_dimensions(self):
        """Test analyzing disparities across multiple dimensions"""
        detector = BiasDetector.__new__(BiasDetector)
        
        metrics = [
            SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
            SliceMetrics("slice2", "Popularity", "Low", 200, 0.8, 0.9, 3.6, 3.7, -0.1, 0.3),
            SliceMetrics("slice3", "Book Era", "Modern", 150, 0.4, 0.5, 3.3, 3.2, 0.1, 0.15),
            SliceMetrics("slice4", "Book Era", "Classic", 120, 0.7, 0.8, 3.8, 3.9, -0.1, 0.25)
        ]
        
        result = detector.analyze_disparities(metrics)
        
        assert "Popularity" in result["summary"]
        assert "Book Era" in result["summary"]
        assert len(result["summary"]) == 2
