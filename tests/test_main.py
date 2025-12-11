import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    """Create a TestClient with queries patched to avoid external calls."""
    # Mock BigQuery access in queries before import side effects
    mock_client = MagicMock()
    mock_client.project = "test-project"
    mock_client.query.return_value.result.return_value = []
    monkeypatch.setattr("api.database.bigquery.Client", lambda: mock_client)

    queries = importlib.reload(importlib.import_module("api.queries"))

    monkeypatch.setattr(queries, "ensure_global_top10_table_exists", lambda: None)
    monkeypatch.setattr(queries, "check_user_exists", lambda user_id: True)
    monkeypatch.setattr(
        queries,
        "get_top_recommendations",
        lambda user_id: [
            {"book_id": 1, "title": "A", "author": "X", "predicted_rating": 4.5}
        ],
    )
    monkeypatch.setattr(
        queries,
        "get_global_top_recommendations",
        lambda: [
            {"book_id": 9, "title": "B", "author": "Y", "predicted_rating": 4.9}
        ],
    )
    monkeypatch.setattr(queries, "log_ctr_event", lambda **kwargs: True)
    monkeypatch.setattr(queries, "get_books_read_by_user", lambda user_id: [{"book_id": 1}])
    monkeypatch.setattr(
        queries, "get_books_not_read_by_user", lambda user_id: [{"book_id": 2}]
    )
    monkeypatch.setattr(queries, "insert_read_interaction", lambda **kwargs: True)

    import api.main as main

    importlib.reload(main)
    return TestClient(main.app)


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_login_existing_user_returns_recommendations(api_client):
    response = api_client.post("/load-recommendation", json={"user_id": "abc"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "abc"
    assert data["recommendations"][0]["title"] == "A"


def test_book_click_failure_returns_500(api_client, monkeypatch):
    monkeypatch.setattr("api.main.log_ctr_event", lambda **kwargs: False)
    response = api_client.post(
        "/book-click",
        json={"user_id": "u1", "book_id": 1, "event_type": "click", "book_title": "A"},
    )
    assert response.status_code == 500


def test_mark_read_failure_returns_500(api_client, monkeypatch):
    monkeypatch.setattr("api.main.insert_read_interaction", lambda **kwargs: False)
    response = api_client.post("/mark-read", json={"user_id": "u", "book_id": 1, "rating": 4})
    assert response.status_code == 500

