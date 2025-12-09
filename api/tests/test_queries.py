import importlib
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _build_mock_client():
    client = MagicMock()
    client.project = "test-project"
    job = MagicMock()
    job.result.return_value = iter([{"cnt": 1}])
    client.query.return_value = job
    client.insert_rows_json.return_value = []
    return client


@pytest.fixture(autouse=True)
def queries_module(monkeypatch):
    """Reload queries with mocked BigQuery client before module side effects run."""
    mock_client = _build_mock_client()
    # Patch database client constructor so queries import does not hit real GCP
    monkeypatch.setattr("api.database.bigquery.Client", lambda: mock_client)
    # Reload queries after patching to ensure ensure_global_top10_table_exists uses mock
    queries = importlib.reload(importlib.import_module("api.queries"))
    return queries, mock_client


def test_check_user_exists_returns_bool(queries_module):
    queries, mock_client = queries_module
    mock_client.query.return_value.result.return_value = iter([{"cnt": 2}])
    assert queries.check_user_exists("user-1") is True


def test_get_top_recommendations_converts_dataframe(monkeypatch, queries_module):
    queries, _ = queries_module
    mock_df = pd.DataFrame(
        [
            {"book_id": 1, "title": "A", "rating": 4.1, "author_names": "X"},
            {"book_id": 2, "title": "B", "rating": 4.0, "author_names": "Y"},
        ]
    )
    mock_generator = MagicMock()
    mock_generator.get_predictions.return_value = mock_df
    monkeypatch.setattr("api.queries.GeneratePredictions", lambda: mock_generator)

    result = queries.get_top_recommendations("user-1")

    assert len(result) == 2
    assert result[0]["author"] == "X"
    assert result[1]["predicted_rating"] == 4.0


def test_get_global_top_recommendations_returns_dicts(queries_module):
    queries, mock_client = queries_module
    mock_client.query.return_value.result.return_value = [
        {"book_id": 10, "predicted_rating": 4.9, "title": "Top", "author": "Auth"}
    ]
    rows = queries.get_global_top_recommendations()
    assert rows == [
        {"book_id": 10, "predicted_rating": 4.9, "title": "Top", "author": "Auth"}
    ]


def test_log_ctr_event_handles_failure(monkeypatch, queries_module):
    queries, _ = queries_module
    logger = MagicMock()
    logger.log_user_event.side_effect = Exception("boom")
    monkeypatch.setattr("api.queries.LogClickEvent", lambda: logger)

    assert queries.log_ctr_event("u", 1, "click") is False


def test_get_books_read_by_user_uses_query(queries_module):
    queries, mock_client = queries_module
    mock_client.query.return_value.result.return_value = [
        {"book_id": 1, "title": "A", "author": "X"}
    ]
    rows = queries.get_books_read_by_user("user-1")
    assert rows[0]["book_id"] == 1


def test_get_books_not_read_by_user(queries_module):
    queries, mock_client = queries_module
    mock_client.query.return_value.result.return_value = [
        {"book_id": 3, "title": "C"}
    ]
    rows = queries.get_books_not_read_by_user("user-1")
    assert rows[0]["title"] == "C"

