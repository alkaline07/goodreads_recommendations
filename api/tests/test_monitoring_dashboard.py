import importlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

import api.monitoring_dashboard as dashboard


def test_ttl_cache_stores_and_expires():
    cache = dashboard.TTLCache(default_ttl_seconds=1)
    cache.set("k", "v")
    assert cache.get("k") == "v"

    cache._cache["k"]["expires_at"] = datetime.utcnow() - timedelta(seconds=1)
    assert cache.get("k") is None


def test_monitoring_bq_client_cache_reuses_client(monkeypatch):
    created = MagicMock(project="test")
    monkeypatch.setattr("api.database.bigquery.Client", lambda: created)
    importlib.reload(dashboard)
    dashboard.MonitoringBQClientCache._instance = None

    client1 = dashboard.MonitoringBQClientCache.get_client()
    client2 = dashboard.MonitoringBQClientCache.get_client()

    assert client1 is client2


def test_run_query_with_timeout_returns_dataframe():
    mock_df = pd.DataFrame([{"a": 1}])

    class QueryJob:
        def result(self, timeout=None):
            class Result:
                def to_dataframe(self_inner):
                    return mock_df

            return Result()

    mock_client = MagicMock()
    mock_client.query.return_value = QueryJob()

    df = dashboard.run_query_with_timeout(mock_client, "SELECT 1")
    assert list(df.to_dict("records")) == [{"a": 1}]


def test_run_query_with_timeout_cancels_on_timeout():
    class QueryJob:
        def __init__(self):
            self.cancelled = False

        def result(self, timeout=None):
            raise TimeoutError("timeout")

        def cancel(self):
            self.cancelled = True

    mock_job = QueryJob()
    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    with pytest.raises(TimeoutError):
        dashboard.run_query_with_timeout(mock_client, "SELECT 1", timeout_seconds=0)

    assert mock_job.cancelled is True

