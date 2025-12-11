import os
from unittest.mock import patch

from api import database


def test_get_bq_client_sets_credentials(monkeypatch):
    """Ensure credentials path is derived from AIRFLOW_HOME and client is created."""
    monkeypatch.setenv("AIRFLOW_HOME", "/tmp/airflow")
    mock_client = object()

    with patch("api.database.bigquery.Client", return_value=mock_client) as mock_ctor:
        client = database.get_bq_client()

    assert client is mock_client
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/airflow/gcp_credentials.json"
    mock_ctor.assert_called_once()

