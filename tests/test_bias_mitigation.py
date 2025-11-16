"""
Test cases for bias_mitigation.py
"""

import pytest
from unittest.mock import Mock, patch
from src.bias_mitigation import BiasMitigator, MitigationConfig, MitigationResult


class TestMitigationConfig:
    """Test MitigationConfig dataclass"""
    
    def test_mitigation_config_creation(self):
        """Test creating MitigationConfig"""
        config = MitigationConfig(
            technique="shrinkage",
            target_dimensions=["Popularity"],
            lambda_shrinkage=0.5
        )
        
        assert config.technique == "shrinkage"
        assert config.lambda_shrinkage == 0.5


class TestBiasMitigator:
    """Test cases for BiasMitigator class"""
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        
        assert mitigator.project_id == "test-project"
        assert mitigator.dataset_id == "books"
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_group_metrics(self, mock_client_class):
        """Test computing group metrics"""
        import pandas as pd
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'slice_group': ['High', 'Low'],
            'count': [100, 200],
            'mean_rating': [3.5, 3.6],
            'std_rating': [0.2, 0.3]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_group_metrics(
            "test-table",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END"
        )
        
        assert isinstance(metrics, dict)
        assert 'High' in metrics
        assert 'Low' in metrics
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_compute_prediction_metrics(self, mock_client_class):
        """Test computing prediction metrics"""
        import pandas as pd
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'count': [100],
            'mae': [0.5],
            'rmse': [0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        metrics = mitigator._compute_prediction_metrics("test-table")
        
        assert metrics['count'] == 100
        assert metrics['mae'] == 0.5
        assert metrics['rmse'] == 0.6
    
    def test_calculate_improvements(self):
        """Test calculating improvements"""
        mitigator = BiasMitigator.__new__(BiasMitigator)
        
        original = {'mae': 0.6, 'rmse': 0.7}
        mitigated = {'mae': 0.5, 'rmse': 0.6}
        
        improvements = mitigator._calculate_improvements(original, mitigated)
        
        assert 'mae_improvement_pct' in improvements
        assert improvements['mae_improvement_pct'] > 0
    
    @patch('src.bias_mitigation.bigquery.Client')
    def test_apply_shrinkage_mitigation(self, mock_client_class):
        """Test applying shrinkage mitigation"""
        mock_client = Mock()
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        
        import pandas as pd
        mock_df = pd.DataFrame({
            'slice_group': ['High'],
            'count': [100],
            'mean_rating': [3.5],
            'std_rating': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mitigator = BiasMitigator()
        mitigator.client = mock_client
        
        result = mitigator.apply_shrinkage_mitigation(
            "input-table",
            "output-table",
            "Popularity",
            "CASE WHEN x > 0.5 THEN 'High' ELSE 'Low' END",
            lambda_shrinkage=0.5
        )
        
        assert isinstance(result, MitigationResult)
        assert result.technique == "shrinkage"

