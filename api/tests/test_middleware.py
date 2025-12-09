import asyncio

import pytest
from starlette.requests import Request
from starlette.responses import Response

from api.middleware import APIMetricsCollector, MonitoringMiddleware


def test_metrics_collector_records_request():
    collector = APIMetricsCollector()
    collector.reset()

    collector.record_request(
        endpoint="/books",
        method="GET",
        status_code=200,
        latency_ms=50.0,
    )

    stats = collector.get_endpoint_stats("GET /books")
    assert stats["request_count"] == 1
    assert stats["avg_latency_ms"] == 50.0
    assert stats["error_count"] == 0


def test_normalize_path_masks_identifiers():
    middleware = MonitoringMiddleware(lambda: None)
    normalized = middleware._normalize_path("/books/12345678901234567890/details")
    assert normalized == "/books/{id}/details"


@pytest.mark.asyncio
async def test_middleware_dispatch_records_metrics():
    async def dummy_app(request: Request) -> Response:
        return Response("ok", status_code=201)

    middleware = MonitoringMiddleware(dummy_app)
    middleware.collector.reset()

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/items/1",
        "raw_path": b"/items/1",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "server": ("test", 80),
    }

    request = Request(scope)

    async def call_next(req: Request) -> Response:
        return await dummy_app(req)

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 201
    stats = middleware.collector.get_endpoint_stats("GET /items/{id}")
    assert stats["request_count"] == 1

