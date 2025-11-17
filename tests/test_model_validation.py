"""
Test cases for model_validation.py
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
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

    @patch("src.model_validation.bigquery.Client")
    def test_init(self, mock_client_class):
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        assert validator.project_id == "test-project"
        assert validator.dataset_id == "books"
        assert "goodreads_train_set" in validator.train_table

    @patch("src.model_validation.bigquery.Client")
    def test_evaluate_split(self, mock_client_class):
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"mean_absolute_error": [0.5], "rmse": [0.6]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        result = validator.evaluate_split("test-model", "val", "test-table")

        assert not result.empty
        assert "mean_absolute_error" in result.columns

    @patch("src.model_validation.bigquery.Client")
    def test_validate_model_pass(self, mock_client_class):
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
        mock_client = Mock()
        mock_client.project = "test-project"

        mock_df = pd.DataFrame({"rmse": [3.5]})
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client

        validator = BigQueryModelValidator()
        validator.client = mock_client

        assert validator.validate_model("test-model", "test-label") is False
