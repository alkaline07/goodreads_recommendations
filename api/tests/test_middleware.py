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

