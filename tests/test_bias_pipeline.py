"""
Comprehensive test cases for bias_pipeline.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
from src.bias_pipeline import BiasAuditPipeline
from src.bias_detection import BiasReport, SliceMetrics
from src.bias_mitigation import MitigationResult


class TestBiasAuditPipeline:
    """Comprehensive test cases for BiasAuditPipeline class"""
    
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
        assert pipeline.detector is not None
        assert pipeline.mitigator is not None
        assert pipeline.visualizer is not None
        assert pipeline.model_selector is not None
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_run_full_audit_no_mitigation(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test running full audit without mitigation"""
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
        assert len(results['mitigation_results']) == 0
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_run_full_audit_with_mitigation(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test running full audit with mitigation"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector_instance.detect_bias.return_value = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": [{"dimension": "Book Era", "severity": "high"}]},
            recommendations=[]
        )
        mock_detector_instance.save_report = Mock()
        mock_detector_instance.create_bias_metrics_table = Mock()
        mock_detector.return_value = mock_detector_instance
        
        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        
        mock_mitigator_instance = Mock()
        mock_mitigator_instance.apply_shrinkage_mitigation = Mock(return_value=None)
        mock_mitigator.return_value = mock_mitigator_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.visualizer = mock_visualizer_instance
        pipeline.mitigator = mock_mitigator_instance
        pipeline.project_id = "test-project"
        pipeline.dataset_id = "books"
        
        # Mock the _apply_prediction_shrinkage method
        with patch.object(pipeline, '_apply_prediction_shrinkage', return_value=MitigationResult(
            technique="prediction_shrinkage",
            timestamp="2025-01-01",
            original_metrics={},
            mitigated_metrics={},
            improvement_pct={},
            output_table="debiased-table"
        )):
            results = pipeline.run_full_audit(
                model_name="test-model",
                predictions_table="test-table",
                apply_mitigation=True,
                mitigation_techniques=['prediction_shrinkage'],
                generate_visualizations=False
            )
            
            assert results['model_name'] == "test-model"
            assert len(results['mitigation_results']) > 0
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_run_full_audit_with_visualizations(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test running full audit with visualizations"""
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
        mock_visualizer_instance.generate_all_visualizations = Mock()
        mock_visualizer.return_value = mock_visualizer_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.visualizer = mock_visualizer_instance
        
        results = pipeline.run_full_audit(
            model_name="test-model",
            predictions_table="test-table",
            apply_mitigation=False,
            generate_visualizations=True
        )
        
        assert results['visualizations_generated'] is True
        mock_visualizer_instance.generate_all_visualizations.assert_called_once()
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_run_full_audit_visualization_exception(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test handling visualization exceptions"""
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
        mock_visualizer_instance.generate_all_visualizations.side_effect = Exception("Viz failed")
        mock_visualizer.return_value = mock_visualizer_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.visualizer = mock_visualizer_instance
        
        results = pipeline.run_full_audit(
            model_name="test-model",
            predictions_table="test-table",
            apply_mitigation=False,
            generate_visualizations=True
        )
        
        assert results['visualizations_generated'] is False
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_validate_mitigation(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test validating mitigation"""
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
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        mitigation_results = [
            MitigationResult(
                technique="shrinkage",
                timestamp="2025-01-01",
                original_metrics={'mae': 0.6},
                mitigated_metrics={'mae': 0.5},
                improvement_pct={'mae_improvement_pct': 16.67},
                output_table="test-table"
            )
        ]
        
        validation = pipeline._validate_mitigation(mitigation_results, "test-model")
        
        assert 'timestamp' in validation
        assert 'techniques_applied' in validation
        assert 'effectiveness' in validation
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_validate_mitigation_with_post_detection(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test validating mitigation with post-mitigation bias detection"""
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
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        mitigation_results = [
            MitigationResult(
                technique="prediction_shrinkage",
                timestamp="2025-01-01",
                original_metrics={'mae': 0.6},
                mitigated_metrics={'mae': 0.5},
                improvement_pct={'mae_improvement_pct': 16.67},
                output_table="test-table"
            )
        ]
        
        validation = pipeline._validate_mitigation(mitigation_results, "test-model")
        
        assert 'effectiveness' in validation
        assert 'prediction_shrinkage' in validation['effectiveness']
        mock_detector_instance.detect_bias.assert_called()
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_validate_mitigation_empty_results(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test validating mitigation with empty results"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        validation = pipeline._validate_mitigation([], "test-model")
        
        assert 'timestamp' in validation
        assert len(validation['techniques_applied']) == 0
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_generate_comprehensive_report(self, mock_makedirs, mock_open, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test generating comprehensive report"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        audit_results = {
            'model_name': 'test-model',
            'timestamp': '2025-01-01',
            'predictions_table': 'test-table',
            'detection_report': BiasReport(
                timestamp="2025-01-01",
                model_name="test-model",
                dataset="test",
                slice_metrics=[],
                disparity_analysis={"detailed_disparities": []},
                recommendations=[]
            ),
            'mitigation_results': [],
            'final_validation': None
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        report_path = pipeline._generate_comprehensive_report(audit_results, "test-model")
        
        assert report_path is not None
        mock_file.write.assert_called()
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_generate_executive_summary_no_bias(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test generating executive summary with no bias"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        audit_results = {
            'detection_report': BiasReport(
                timestamp="2025-01-01",
                model_name="test-model",
                dataset="test",
                slice_metrics=[],
                disparity_analysis={"detailed_disparities": []},
                recommendations=[]
            ),
            'mitigation_results': []
        }
        
        summary = pipeline._generate_executive_summary(audit_results)
        
        assert summary['bias_detected'] is False
        assert summary['severity'] == 'none'
        assert summary['overall_status'] == 'PASS'
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_generate_executive_summary_with_bias(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test generating executive summary with bias"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        audit_results = {
            'detection_report': BiasReport(
                timestamp="2025-01-01",
                model_name="test-model",
                dataset="test",
                slice_metrics=[],
                disparity_analysis={"detailed_disparities": [{"severity": "high", "dimension": "Popularity"}]},
                recommendations=[]
            ),
            'mitigation_results': []
        }
        
        summary = pipeline._generate_executive_summary(audit_results)
        
        assert summary['bias_detected'] is True
        assert summary['severity'] == 'high'
        assert summary['overall_status'] == 'NEEDS_ATTENTION'
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_generate_executive_summary_mitigated(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test generating executive summary with mitigated bias"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        
        audit_results = {
            'detection_report': BiasReport(
                timestamp="2025-01-01",
                model_name="test-model",
                dataset="test",
                slice_metrics=[],
                disparity_analysis={"detailed_disparities": [{"severity": "high", "dimension": "Popularity"}]},
                recommendations=[]
            ),
            'mitigation_results': [MitigationResult(
                technique="shrinkage",
                timestamp="2025-01-01",
                original_metrics={},
                mitigated_metrics={},
                improvement_pct={},
                output_table="test-table"
            )]
        }
        
        summary = pipeline._generate_executive_summary(audit_results)
        
        assert summary['bias_detected'] is True
        assert summary['mitigation_applied'] is True
        assert summary['overall_status'] == 'MITIGATED'
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    @patch('google.cloud.bigquery.Client')
    def test_apply_prediction_shrinkage(self, mock_bq_client, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test applying prediction shrinkage"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        mock_mae_df = pd.DataFrame({'mae': [0.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_mae_df
        mock_bq_client.return_value = mock_client
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.project_id = "test-project"
        pipeline.dataset_id = "books"
        
        detection_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": [{"dimension": "Book Era", "mae_cv": 0.3}]},
            recommendations=[]
        )
        
        result = pipeline._apply_prediction_shrinkage(
            "test-table",
            "test-model",
            detection_report
        )
        
        assert result is not None
        assert result.technique == "prediction_shrinkage"
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_apply_prediction_shrinkage_no_era_bias(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test applying prediction shrinkage when no Book Era bias"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.project_id = "test-project"
        pipeline.dataset_id = "books"
        
        detection_report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={"detailed_disparities": [{"dimension": "Popularity", "mae_cv": 0.3}]},
            recommendations=[]
        )
        
        result = pipeline._apply_prediction_shrinkage(
            "test-table",
            "test-model",
            detection_report
        )
        
        assert result is None
    
    @patch('src.bias_pipeline.BiasDetector')
    @patch('src.bias_pipeline.BiasMitigator')
    @patch('src.bias_pipeline.BiasVisualizer')
    @patch('src.bias_pipeline.ModelSelector')
    def test_get_table_mae(self, mock_selector, mock_visualizer, mock_mitigator, mock_detector):
        """Test getting table MAE"""
        mock_detector_instance = Mock()
        mock_detector_instance.project_id = "test-project"
        mock_detector.return_value = mock_detector_instance
        
        pipeline = BiasAuditPipeline()
        pipeline.detector = mock_detector_instance
        pipeline.project_id = "test-project"
        
        with patch('google.cloud.bigquery.Client') as mock_bq:
            mock_client = Mock()
            mock_df = pd.DataFrame({'mae': [0.5]})
            mock_client.query.return_value.to_dataframe.return_value = mock_df
            mock_bq.return_value = mock_client
            
            mae = pipeline._get_table_mae("test-table")
            assert mae == 0.5
