"""
Test cases for bias_pipeline.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.bias_pipeline import BiasAuditPipeline


class TestBiasAuditPipeline:
    """Test cases for BiasAuditPipeline class"""
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_init(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test initialization"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        
        assert pipeline.project_id == "test-project"
        assert pipeline.dataset_id == "books"
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_run_full_audit(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test running full audit"""
        from src.bias_detection import BiasReport
        
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector_instance.detect_bias.return_value = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": []},
            recommendations=[]
        )
        mock_detector_instance.save_report = Mock()
        mock_detector_instance.create_bias_metrics_table = Mock()
        mock_detector.return_value = mock_detector_instance
        
        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.visualizer = mock_visualizer_instance
        
        results = pipeline.run_full_audit(
            model_name="test-model",
            predictions_table="test-table",
            apply_mitigation=False,
            generate_visualizations=False
        )
        
        assert results['model_name'] == "test-model"
        assert results['detection_report'] is not None
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_validate_mitigation(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test validating mitigation"""
        from src.bias_mitigation import MitigationResult
        
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        mitigation_results = [
            MitigationResult(
                technique="shrinkage",
                timestamp="2025-01-01",
                original_metrics={},
                mitigated_metrics={},
                improvement_pct={},
                output_table="test-table"
            )
        ]
        
        validation = pipeline._validate_mitigation(mitigation_results, "test-model")
        
        assert 'timestamp' in validation
        assert 'techniques_applied' in validation

