"""
Comprehensive test cases for model_manager.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import json
import os
from datetime import datetime
from src.model_manager import (
    ModelManager,
    safe_mlflow_log,
    get_selected_model_from_report,
    get_vertex_ai_model_name
)


class TestSafeMlflowLog:
    """Test safe_mlflow_log function"""
    
    def test_safe_mlflow_log_success(self):
        """Test successful MLflow logging"""
        mock_func = Mock(return_value="success")
        result = safe_mlflow_log(mock_func, "arg1", "arg2", key="value")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", key="value")
    
    def test_safe_mlflow_log_exception(self):
        """Test MLflow logging with exception"""
        mock_func = Mock(side_effect=Exception("MLflow error"))
        result = safe_mlflow_log(mock_func, "arg1")
        
        assert result is None
        mock_func.assert_called_once_with("arg1")


class TestModelManager:
    """Comprehensive test cases for ModelManager class"""
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_init_with_airflow_home(self, mock_client_class, mock_mlflow_uri, 
                                     mock_mlflow_exp, mock_aiplatform_init):
        """Test initialization with AIRFLOW_HOME set"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'}):
            manager = ModelManager()
            
            assert manager.project_id == "test-project"
            assert manager.vertex_region == "us-central1"
            assert manager.improvement_threshold == 0.0
            mock_aiplatform_init.assert_called_once()
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class, mock_mlflow_uri,
                                       mock_mlflow_exp, mock_aiplatform_init):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            manager = ModelManager()
            
            assert manager.project_id == "test-project"
            assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_init_with_custom_params(self, mock_client_class, mock_mlflow_uri,
                                     mock_mlflow_exp, mock_aiplatform_init):
        """Test initialization with custom parameters"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client
        
        manager = ModelManager(
            project_id="custom-project",
            vertex_region="us-east1",
            improvement_threshold=0.02
        )
        
        assert manager.project_id == "custom-project"
        assert manager.vertex_region == "us-east1"
        assert manager.improvement_threshold == 0.02
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_init_mlflow_exception(self, mock_client_class, mock_mlflow_uri,
                                   mock_mlflow_exp, mock_aiplatform_init):
        """Test initialization when MLflow fails"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        mock_mlflow_uri.side_effect = Exception("MLflow error")
        
        # Should not raise exception
        manager = ModelManager()
        assert manager.project_id == "test-project"
    
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_versions_success(self, mock_client_class, mock_mlflow_uri,
                                       mock_mlflow_exp, mock_aiplatform_init,
                                       mock_model_list):
        """Test getting model versions successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        # Create mock version objects
        mock_version1 = Mock()
        mock_version1.version_id = "2"
        mock_version1.version_aliases = ["default"]
        mock_version2 = Mock()
        mock_version2.version_id = "1"
        mock_version2.version_aliases = []
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_version1, mock_version2]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        
        manager = ModelManager()
        versions = manager.get_model_versions("test-model")
        
        assert len(versions) == 2
        assert versions[0].version_id == "2"
        mock_model_list.assert_called_once()
    
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_versions_empty(self, mock_client_class, mock_mlflow_uri,
                                      mock_mlflow_exp, mock_aiplatform_init,
                                      mock_model_list):
        """Test getting model versions when none exist"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        mock_model_list.return_value = []
        
        manager = ModelManager()
        versions = manager.get_model_versions("test-model")
        
        assert len(versions) == 0
    
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_versions_exception(self, mock_client_class, mock_mlflow_uri,
                                          mock_mlflow_exp, mock_aiplatform_init,
                                          mock_model_list):
        """Test getting model versions with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        mock_model_list.side_effect = Exception("List failed")
        
        manager = ModelManager()
        versions = manager.get_model_versions("test-model")
        
        assert len(versions) == 0
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_metrics_from_bq_success(self, mock_client_class, mock_mlflow_uri,
                                               mock_mlflow_exp, mock_aiplatform_init):
        """Test getting model metrics from BigQuery successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        metrics = manager.get_model_metrics_from_bq("project.dataset.table")
        
        assert metrics is not None
        assert metrics['num_predictions'] == 1000
        assert metrics['mae'] == 0.5
        assert metrics['rmse'] == 0.6
        assert metrics['r_squared'] == 0.64
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_metrics_from_bq_empty(self, mock_client_class, mock_mlflow_uri,
                                            mock_mlflow_exp, mock_aiplatform_init):
        """Test getting model metrics with empty result"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame()
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        metrics = manager.get_model_metrics_from_bq("project.dataset.table")
        
        assert metrics is None
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_metrics_from_bq_nan_values(self, mock_client_class, mock_mlflow_uri,
                                                  mock_mlflow_exp, mock_aiplatform_init):
        """Test getting model metrics with NaN values"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [None],
            'r_squared': [None],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [None]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        metrics = manager.get_model_metrics_from_bq("project.dataset.table")
        
        assert metrics is not None
        assert metrics['r_squared'] == 0.0
        assert metrics['correlation'] == 0.0
        assert metrics['std_error'] == 0.0
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_get_model_metrics_from_bq_exception(self, mock_client_class, mock_mlflow_uri,
                                                 mock_mlflow_exp, mock_aiplatform_init):
        """Test getting model metrics with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        metrics = manager.get_model_metrics_from_bq("project.dataset.table")
        
        assert metrics is None
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_compare_models_promote(self, mock_client_class, mock_mlflow_uri,
                                    mock_mlflow_exp, mock_aiplatform_init):
        """Test comparing models when new model should be promoted"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        manager = ModelManager(improvement_threshold=0.0)
        
        new_metrics = {'rmse': 0.5, 'mae': 0.4, 'r_squared': 0.8}
        current_metrics = {'rmse': 0.6, 'mae': 0.5, 'r_squared': 0.7}
        
        should_promote, improvements = manager.compare_models(new_metrics, current_metrics)
        
        assert should_promote is True
        assert improvements['rmse_improvement'] > 0
        assert improvements['r_squared_improvement'] > 0
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_compare_models_rollback(self, mock_client_class, mock_mlflow_uri,
                                     mock_mlflow_exp, mock_aiplatform_init):
        """Test comparing models when new model should not be promoted"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        manager = ModelManager(improvement_threshold=0.0)
        
        new_metrics = {'rmse': 0.7, 'mae': 0.6, 'r_squared': 0.6}
        current_metrics = {'rmse': 0.5, 'mae': 0.4, 'r_squared': 0.8}
        
        should_promote, improvements = manager.compare_models(new_metrics, current_metrics)
        
        assert should_promote is False
        assert improvements['rmse_improvement'] < 0
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_compare_models_with_threshold(self, mock_client_class, mock_mlflow_uri,
                                           mock_mlflow_exp, mock_aiplatform_init):
        """Test comparing models with improvement threshold"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        manager = ModelManager(improvement_threshold=0.1)  # 10% improvement required
        
        new_metrics = {'rmse': 0.55, 'mae': 0.45, 'r_squared': 0.75}
        current_metrics = {'rmse': 0.6, 'mae': 0.5, 'r_squared': 0.7}
        
        should_promote, improvements = manager.compare_models(new_metrics, current_metrics)
        
        # 0.55 vs 0.6 = 8.33% improvement, less than 10% threshold
        assert should_promote is False
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_compare_models_degradation_check(self, mock_client_class, mock_mlflow_uri,
                                              mock_mlflow_exp, mock_aiplatform_init):
        """Test comparing models with degradation in secondary metrics"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        manager = ModelManager(improvement_threshold=0.0)
        
        # RMSE improves but MAE degrades significantly
        new_metrics = {'rmse': 0.4, 'mae': 0.6, 'r_squared': 0.5}
        current_metrics = {'rmse': 0.5, 'mae': 0.4, 'r_squared': 0.8}
        
        should_promote, improvements = manager.compare_models(new_metrics, current_metrics)
        
        # Should not promote due to degradation check
        assert should_promote is False
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_find_latest_bqml_model_success(self, mock_client_class, mock_mlflow_uri,
                                           mock_mlflow_exp, mock_aiplatform_init):
        """Test finding latest BQML model successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        
        mock_model1 = Mock()
        mock_model1.model_id = "model_v2"
        mock_model1.created = Mock()
        mock_model1.created.timestamp.return_value = 2000.0
        
        mock_model2 = Mock()
        mock_model2.model_id = "model_v1"
        mock_model2.created = Mock()
        mock_model2.created.timestamp.return_value = 1000.0
        
        mock_dataset = Mock()
        mock_client.dataset.return_value = mock_dataset
        mock_client.list_models.return_value = [mock_model1, mock_model2]
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        model_id = manager.find_latest_bqml_model("test_dataset", "model_")
        
        assert model_id == "test-project.test_dataset.model_v2"
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_find_latest_bqml_model_not_found(self, mock_client_class, mock_mlflow_uri,
                                              mock_mlflow_exp, mock_aiplatform_init):
        """Test finding latest BQML model when none match prefix"""
        mock_client = Mock()
        mock_client.project = "test-project"
        
        mock_model = Mock()
        mock_model.model_id = "other_model"
        
        mock_dataset = Mock()
        mock_client.dataset.return_value = mock_dataset
        mock_client.list_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        model_id = manager.find_latest_bqml_model("test_dataset", "model_")
        
        assert model_id is None
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_find_latest_bqml_model_exception(self, mock_client_class, mock_mlflow_uri,
                                              mock_mlflow_exp, mock_aiplatform_init):
        """Test finding latest BQML model with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.dataset.side_effect = Exception("Dataset error")
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        model_id = manager.find_latest_bqml_model("test_dataset", "model_")
        
        assert model_id is None
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_generate_predictions_table_success(self, mock_client_class, mock_mlflow_uri,
                                                mock_mlflow_exp, mock_aiplatform_init):
        """Test generating predictions table successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_client.query.return_value = mock_job
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        result = manager.generate_predictions_table(
            "project.dataset.model",
            "project.dataset.val_table",
            "project.dataset.output_table",
            sample_size=1000
        )
        
        assert result is True
        mock_client.query.assert_called_once()
    
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_generate_predictions_table_exception(self, mock_client_class, mock_mlflow_uri,
                                                   mock_mlflow_exp, mock_aiplatform_init):
        """Test generating predictions table with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client
        
        manager = ModelManager()
        result = manager.generate_predictions_table(
            "project.dataset.model",
            "project.dataset.val_table",
            "project.dataset.output_table"
        )
        
        assert result is False
    
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_set_default_version_success(self, mock_client_class, mock_mlflow_uri,
                                        mock_mlflow_exp, mock_aiplatform_init,
                                        mock_model_list):
        """Test setting default version successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_old_version = Mock()
        mock_old_version.version_id = "1"
        mock_old_version.version_aliases = ["default"]
        mock_old_version.remove_version_aliases = Mock()
        
        mock_new_version = Mock()
        mock_new_version.version_id = "2"
        mock_new_version.version_aliases = []  # Empty list, not a Mock
        mock_new_version.add_version_aliases = Mock()
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_new_version, mock_old_version]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        
        manager = ModelManager()
        manager.set_default_version(mock_new_version, "test-model")
        
        mock_old_version.remove_version_aliases.assert_called_once_with(['default'])
        mock_new_version.add_version_aliases.assert_called_once_with(['default'])
    
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_set_default_version_exception(self, mock_client_class, mock_mlflow_uri,
                                           mock_mlflow_exp, mock_aiplatform_init,
                                           mock_model_list):
        """Test setting default version with exception"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_model = Mock()
        mock_model.version_id = "1"
        mock_model.add_version_aliases.side_effect = Exception("Add alias failed")
        mock_model_list.return_value = [mock_model]
        
        manager = ModelManager()
        
        with pytest.raises(Exception):
            manager.set_default_version(mock_model, "test-model")
    
    @patch('src.model_manager.mlflow.start_run')
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_manage_model_rollback_no_versions(self, mock_client_class, mock_mlflow_uri,
                                               mock_mlflow_exp, mock_aiplatform_init,
                                               mock_model_list, mock_mlflow_run):
        """Test rollback management with no model versions"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        mock_model_list.return_value = []
        mock_mlflow_run.return_value.__enter__.return_value = None
        
        manager = ModelManager()
        result = manager.manage_model_rollback(
            "test-model",
            "project.dataset.new_predictions"
        )
        
        assert result['decision'] == 'ERROR_NO_VERSIONS'
    
    @patch('src.model_manager.mlflow.start_run')
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_manage_model_rollback_first_deployment(self, mock_client_class, mock_mlflow_uri,
                                                    mock_mlflow_exp, mock_aiplatform_init,
                                                    mock_model_list, mock_mlflow_run):
        """Test rollback management for first deployment"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_model = Mock()
        mock_model.version_id = "1"
        mock_model.version_aliases = []
        mock_model.add_version_aliases = Mock()
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_model]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        mock_mlflow_run.return_value.__enter__.return_value = None
        
        manager = ModelManager()
        manager.set_default_version = Mock()
        
        result = manager.manage_model_rollback(
            "test-model",
            "project.dataset.new_predictions"
        )
        
        assert result['decision'] == 'PROMOTED_FIRST_DEFAULT'
        manager.set_default_version.assert_called_once()
    
    @patch('src.model_manager.mlflow.start_run')
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_manage_model_rollback_promote_new(self, mock_client_class, mock_mlflow_uri,
                                               mock_mlflow_exp, mock_aiplatform_init,
                                               mock_model_list, mock_mlflow_run):
        """Test rollback management promoting new model"""
        mock_client = Mock()
        mock_client.project = "test-project"
        
        new_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.4],
            'rmse': [0.5],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        
        current_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.7],
            'r_squared': [0.49],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        
        mock_client.query.return_value.to_dataframe.side_effect = [new_df, current_df]
        mock_client_class.return_value = mock_client
        
        mock_new_model = Mock()
        mock_new_model.version_id = "2"
        mock_new_model.version_aliases = []
        
        mock_current_model = Mock()
        mock_current_model.version_id = "1"
        mock_current_model.version_aliases = ["default"]
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_new_model, mock_current_model]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        mock_mlflow_run.return_value.__enter__.return_value = None
        
        manager = ModelManager()
        manager.set_default_version = Mock()
        
        result = manager.manage_model_rollback(
            "test-model",
            "project.dataset.new_predictions",
            "project.dataset.current_predictions"
        )
        
        assert result['decision'] == 'PROMOTED_NEW_VERSION'
        manager.set_default_version.assert_called_once()
    
    @patch('src.model_manager.mlflow.start_run')
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_manage_model_rollback_keep_current(self, mock_client_class, mock_mlflow_uri,
                                                mock_mlflow_exp, mock_aiplatform_init,
                                                mock_model_list, mock_mlflow_run):
        """Test rollback management keeping current model"""
        mock_client = Mock()
        mock_client.project = "test-project"
        
        new_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.6],
            'rmse': [0.7],
            'correlation': [0.6],
            'r_squared': [0.36],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        
        current_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        
        mock_client.query.return_value.to_dataframe.side_effect = [new_df, current_df]
        mock_client_class.return_value = mock_client
        
        mock_new_model = Mock()
        mock_new_model.version_id = "2"
        mock_new_model.version_aliases = []
        
        mock_current_model = Mock()
        mock_current_model.version_id = "1"
        mock_current_model.version_aliases = ["default"]
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_new_model, mock_current_model]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        mock_mlflow_run.return_value.__enter__.return_value = None
        
        manager = ModelManager()
        
        result = manager.manage_model_rollback(
            "test-model",
            "project.dataset.new_predictions",
            "project.dataset.current_predictions"
        )
        
        assert result['decision'] == 'ROLLBACK_KEPT_CURRENT'
    
    @patch('src.model_manager.mlflow.start_run')
    @patch('src.model_manager.aiplatform.Model.list')
    @patch('src.model_manager.aiplatform.init')
    @patch('src.model_manager.mlflow.set_experiment')
    @patch('src.model_manager.mlflow.set_tracking_uri')
    @patch('src.model_manager.bigquery.Client')
    def test_manage_model_rollback_no_comparison(self, mock_client_class, mock_mlflow_uri,
                                                 mock_mlflow_exp, mock_aiplatform_init,
                                                 mock_model_list, mock_mlflow_run):
        """Test rollback management without current predictions table"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({
            'num_predictions': [1000],
            'mae': [0.5],
            'rmse': [0.6],
            'correlation': [0.8],
            'r_squared': [0.64],
            'mean_predicted': [3.5],
            'mean_actual': [3.6],
            'std_error': [0.2]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        mock_new_model = Mock()
        mock_new_model.version_id = "2"
        mock_new_model.version_aliases = []
        
        mock_current_model = Mock()
        mock_current_model.version_id = "1"
        mock_current_model.version_aliases = ["default"]
        
        # Create mock parent model with versioning_registry
        mock_parent_model = Mock()
        mock_versioning_registry = Mock()
        mock_versioning_registry.list_versions.return_value = [mock_new_model, mock_current_model]
        mock_parent_model.versioning_registry = mock_versioning_registry
        
        # Model.list() returns list of parent models
        mock_model_list.return_value = [mock_parent_model]
        mock_mlflow_run.return_value.__enter__.return_value = None
        
        manager = ModelManager()
        
        result = manager.manage_model_rollback(
            "test-model",
            "project.dataset.new_predictions",
            current_model_predictions_table=None
        )
        
        assert result['decision'] == 'KEPT_CURRENT_NO_COMPARISON'


class TestHelperFunctions:
    """Test helper functions"""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"selected_model": {"model_name": "test_model", "predictions_table": "test_table"}}')
    def test_get_selected_model_from_report_success(self, mock_file):
        """Test getting selected model from report successfully"""
        with patch('os.path.exists', return_value=True):
            result = get_selected_model_from_report()
            
            assert result is not None
            assert result['model_name'] == "test_model"
            assert result['predictions_table'] == "test_table"
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_get_selected_model_from_report_not_found(self, mock_file):
        """Test getting selected model when report not found"""
        result = get_selected_model_from_report()
        
        assert result is None
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json content')
    def test_get_selected_model_from_report_invalid_json(self, mock_file):
        """Test getting selected model with invalid JSON"""
        with patch('os.path.exists', return_value=True):
            # Function catches exceptions and returns None
            result = get_selected_model_from_report()
            assert result is None
    
    def test_get_vertex_ai_model_name_boosted_tree(self):
        """Test mapping boosted_tree_regressor to Vertex AI name"""
        result = get_vertex_ai_model_name("boosted_tree_regressor")
        assert result == "goodreads_boosted_tree_regressor"
    
    def test_get_vertex_ai_model_name_matrix_factorization(self):
        """Test mapping matrix_factorization to Vertex AI name"""
        result = get_vertex_ai_model_name("matrix_factorization")
        assert result == "goodreads_matrix_factorization"
    
    def test_get_vertex_ai_model_name_unknown(self):
        """Test mapping unknown model name"""
        result = get_vertex_ai_model_name("unknown_model")
        assert result == "goodreads_unknown_model"
    
    def test_get_vertex_ai_model_name_default_prefix(self):
        """Test default prefix for unmapped models"""
        result = get_vertex_ai_model_name("custom_model")
        assert result == "goodreads_custom_model"

