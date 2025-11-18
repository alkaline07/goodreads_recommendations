"""
Comprehensive test cases for model_validation.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json
import os
from src.model_validation import BigQueryModelValidator


@pytest.fixture(autouse=True)
def mock_mlflow_all():
    """Mock MLflow globally to avoid network/403 errors."""
    with patch("src.model_validation.mlflow.set_tracking_uri"), \
         patch("src.model_validation.mlflow.set_experiment"), \
         patch("src.model_validation.mlflow.log_metric"), \
         patch("src.model_validation.mlflow.log_param"), \
         patch("src.model_validation.mlflow.log_artifact"), \
         patch("src.model_validation.mlflow.start_run") as mock_run:

        mock_context = Mock()
        mock_run.return_value.__enter__.return_value = mock_context
        mock_run.return_value.__exit__.return_value = None
        yield mock_run


class TestBigQueryModelValidator:
    """Comprehensive test cases for BigQueryModelValidator class"""
    
    @patch("src.model_validation.bigquery.Client")
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        assert validator.project_id == "test-project"
        assert validator.dataset_id == "books"
        assert "goodreads_train_set" in validator.train_table
        assert "goodreads_validation_set" in validator.val_table
        assert "goodreads_test_set" in validator.test_table
    
    @patch("src.model_validation.bigquery.Client")
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        with patch.dict('os.environ', {}, clear=True):
            validator = BigQueryModelValidator()
            assert validator.project_id == "test-project"
    
    @patch("src.model_validation.bigquery.Client")
    def test_init_with_custom_project_id(self, mock_client_class):
        """Test initialization with custom project_id"""
        mock_client = Mock()
        mock_client.project = "custom-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator(project_id="custom-project")
        assert validator.project_id == "custom-project"
    
    @patch("src.model_validation.bigquery.Client")
    def test_init_with_custom_dataset_id(self, mock_client_class):
        """Test initialization with custom dataset_id"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator(dataset_id="custom_dataset")
        assert validator.dataset_id == "custom_dataset"
    
    @patch("src.model_validation.bigquery.Client")
    def test_evaluate_split_success(self, mock_client_class):
        """Test evaluating split successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({
            "mean_absolute_error": [0.5],
            "rmse": [0.6],
            "r2_score": [0.8]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        result = validator.evaluate_split("test-model", "val", "test-table")

        assert not result.empty
        assert "mean_absolute_error" in result.columns
        assert "rmse" in result.columns
    
    @patch("src.model_validation.bigquery.Client")
    def test_evaluate_split_empty_result(self, mock_client_class):
        """Test evaluating split with empty result"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame()
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        result = validator.evaluate_split("test-model", "val", "test-table")

        assert result.empty
    
    @patch("src.model_validation.bigquery.Client")
    def test_evaluate_split_exception(self, mock_client_class):
        """Test exception handling in evaluate_split"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("Query failed")
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        with pytest.raises(Exception):
            validator.evaluate_split("test-model", "val", "test-table")
    
    @patch("src.model_validation.bigquery.Client")
    def test_evaluate_split_all_splits(self, mock_client_class):
        """Test evaluating all splits (train, val, test)"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({
            "mean_absolute_error": [0.5],
            "rmse": [0.6]
        })
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        for split in ["train", "val", "test"]:
            result = validator.evaluate_split("test-model", split, f"test-{split}-table")
            assert not result.empty
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_pass(self, mock_client_class):
        """Test validating model that passes threshold"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"rmse": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is True
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_fail(self, mock_client_class):
        """Test validating model that fails threshold"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"rmse": [3.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is False
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_exactly_at_threshold(self, mock_client_class):
        """Test validating model exactly at threshold"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"rmse": [3.0]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is True
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_with_root_mean_squared_error_column(self, mock_client_class):
        """Test validating model with root_mean_squared_error column name"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"root_mean_squared_error": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is True
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_with_mean_squared_error_column(self, mock_client_class):
        """Test validating model with mean_squared_error column name"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"mean_squared_error": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is True
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_missing_rmse_column(self, mock_client_class):
        """Test validating model when RMSE column is missing"""
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"other_metric": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        with pytest.raises(KeyError):
            validator.validate_model("test-model", "test-label")
    
    @patch("src.model_validation.bigquery.Client")
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_json_report(self, mock_makedirs, mock_open, mock_client_class):
        """Test saving JSON report"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        results = {
            "train": pd.DataFrame({"mae": [0.5]}),
            "val": pd.DataFrame({"mae": [0.6]}),
            "test": pd.DataFrame({"mae": [0.7]})
        }

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        validator.save_json_report("test-label", "test-model", results)

        mock_makedirs.assert_called()
        mock_file.write.assert_called()
        # Collect all write calls and join them to reconstruct the JSON
        written_content = ''.join([call[0][0] for call in mock_file.write.call_args_list])
        written_data = json.loads(written_content)
        assert written_data["model_label"] == "test-label"
    
    @patch("src.model_validation.bigquery.Client")
    def test_get_selected_model_from_report_success(self, mock_client_class):
        """Test getting selected model from report successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        report_data = {
            "selected_model": {
                "model_name": "boosted_tree_regressor",
                "predictions_table": "test-project.books.boosted_tree_rating_predictions"
            }
        }

        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(report_data)
            mock_open.return_value.__enter__.return_value = mock_file

            result = validator.get_selected_model_from_report()

            assert result is not None
            assert result['model_name'] == "boosted_tree_regressor"
    
    @patch("src.model_validation.bigquery.Client")
    def test_get_selected_model_from_report_not_found(self, mock_client_class):
        """Test getting selected model when report doesn't exist"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = validator.get_selected_model_from_report()

            assert result is None
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_selected_model_with_report(self, mock_client_class):
        """Test validating selected model from report"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({"rmse": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        with patch.object(validator, 'get_selected_model_from_report', return_value={
            'model_name': 'boosted_tree_regressor',
            'predictions_table': 'test-table'
        }):
            with patch.object(validator, 'validate_model', return_value=True):
                result = validator.validate_selected_model()
                assert result is True
    
    @patch("src.model_validation.bigquery.Client")
    def test_validate_selected_model_without_report(self, mock_client_class):
        """Test validating selected model without report (uses default)"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_df = pd.DataFrame({"rmse": [2.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        with patch.object(validator, 'get_selected_model_from_report', return_value=None):
            with patch.object(validator, 'validate_model', return_value=True):
                result = validator.validate_selected_model()
                assert result is True
