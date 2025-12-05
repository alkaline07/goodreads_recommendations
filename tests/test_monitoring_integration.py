"""
Integration Smoke Tests for Monitoring System

Verifies:
1. All monitoring modules can be imported
2. No circular dependency issues
3. Core functionality initializes correctly
4. API endpoints are accessible

Author: Goodreads Recommendation Team
Date: 2025
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import sys


class TestMonitoringImports:
    """Test that all monitoring modules can be imported without errors."""
    
    def test_model_monitoring_import(self):
        """Verify model monitoring module imports successfully."""
        try:
            from src.model_monitoring import ModelMonitor, run_full_monitoring
            assert ModelMonitor is not None
            assert run_full_monitoring is not None
        except ImportError as e:
            pytest.fail(f"Failed to import model_monitoring: {e}")
    
    def test_api_middleware_import(self):
        """Verify API middleware imports successfully."""
        try:
            from api.middleware import MonitoringMiddleware, APIMetricsCollector, get_metrics_collector
            assert MonitoringMiddleware is not None
            assert APIMetricsCollector is not None
            assert get_metrics_collector is not None
        except ImportError as e:
            pytest.fail(f"Failed to import middleware: {e}")
    
    def test_monitoring_dashboard_import(self):
        """Verify monitoring dashboard imports successfully."""
        try:
            from api.monitoring_dashboard import router
            assert router is not None
        except ImportError as e:
            pytest.fail(f"Failed to import monitoring_dashboard: {e}")


class TestAPIMetricsCollector:
    """Test API metrics collector functionality."""
    
    def test_singleton_pattern(self):
        """Verify APIMetricsCollector is a singleton."""
        from api.middleware import APIMetricsCollector
        
        collector1 = APIMetricsCollector()
        collector2 = APIMetricsCollector()
        
        assert collector1 is collector2, "APIMetricsCollector should be a singleton"
    
    def test_record_request(self):
        """Verify request recording works."""
        from api.middleware import APIMetricsCollector
        
        collector = APIMetricsCollector()
        collector.reset()
        
        collector.record_request(
            endpoint="/test-endpoint",
            method="GET",
            status_code=200,
            latency_ms=150.0
        )
        
        stats = collector.get_endpoint_stats("GET /test-endpoint")
        
        assert stats['request_count'] == 1
        assert stats['avg_latency_ms'] == 150.0
        assert stats['error_count'] == 0
    
    def test_error_tracking(self):
        """Verify error tracking works."""
        from api.middleware import APIMetricsCollector
        
        collector = APIMetricsCollector()
        collector.reset()
        
        collector.record_request(
            endpoint="/error-endpoint",
            method="POST",
            status_code=500,
            latency_ms=200.0,
            error_message="Internal server error"
        )
        
        stats = collector.get_endpoint_stats("POST /error-endpoint")
        
        assert stats['error_count'] == 1
        assert stats['error_rate'] == 100.0
    
    def test_slow_request_detection(self):
        """Verify slow request detection."""
        from api.middleware import APIMetricsCollector
        
        collector = APIMetricsCollector()
        collector.reset()
        
        collector.record_request(
            endpoint="/slow-endpoint",
            method="GET",
            status_code=200,
            latency_ms=1500.0
        )
        
        stats = collector.get_endpoint_stats("GET /slow-endpoint")
        
        assert stats['slow_requests'] == 1


class TestModelMonitorInitialization:
    """Test model monitor initialization."""
    
    @patch('src.model_monitoring.bigquery.Client')
    @patch('src.model_monitoring.mlflow')
    def test_model_monitor_init(self, mock_mlflow, mock_bq_client):
        """Verify ModelMonitor initializes without errors."""
        from src.model_monitoring import ModelMonitor
        
        mock_bq_client.return_value.project = "test-project"
        mock_bq_client.return_value.query.return_value.result.return_value = None
        
        monitor = ModelMonitor(project_id="test-project")
        
        assert monitor.project_id == "test-project"
        assert monitor.dataset_id == "books"
        assert "model_metrics_history" in monitor.metrics_table
        assert "data_drift_history" in monitor.drift_table


class TestAPIEndpointsExist:
    """Verify all monitoring API endpoints are registered."""
    
    def test_api_app_has_monitoring_routes(self):
        """Verify FastAPI app has monitoring routes registered."""
        from api.main import app
        
        routes = [route.path for route in app.routes]
        
        assert "/metrics" in routes, "Missing /metrics endpoint"
        assert "/metrics/timeline" in routes, "Missing /metrics/timeline endpoint"
        assert "/frontend-metrics" in routes, "Missing /frontend-metrics endpoint"
        assert "/metrics/frontend" in routes, "Missing /metrics/frontend endpoint"
    
    def test_monitoring_dashboard_routes(self):
        """Verify monitoring dashboard routes are registered."""
        from api.main import app
        
        routes = [route.path for route in app.routes]
        
        monitoring_routes = [r for r in routes if r.startswith('/report')]
        
        assert len(monitoring_routes) > 0, "No /report routes found"
        assert "/report" in routes or "/report/" in routes, "Missing /report dashboard route"


class TestMiddlewareIntegration:
    """Test that middleware integrates correctly with FastAPI."""
    
    def test_app_has_middleware(self):
        """Verify app has monitoring middleware registered."""
        from api.main import app
        
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        
        assert "MonitoringMiddleware" in middleware_classes, "MonitoringMiddleware not registered"
        assert "CORSMiddleware" in middleware_classes, "CORSMiddleware not registered"


class TestDAGIntegration:
    """Test DAG integration with monitoring."""
    
    def test_dag_has_monitoring_tasks(self):
        """Verify DAG has monitoring tasks defined."""
        import sys
        import os
        
        dag_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'dags', 'data_pipeline_dag.py')
        
        with open(dag_path, 'r') as f:
            dag_content = f.read()
        
        assert 'model_monitoring_run' in dag_content, "Missing model_monitoring_run function"
        assert 'data_drift_check_run' in dag_content, "Missing data_drift_check_run function"
        assert 'model_monitoring_task' in dag_content, "Missing model_monitoring_task"
        assert 'data_drift_check_task' in dag_content, "Missing data_drift_check_task"
        assert 'from src.model_monitoring import' in dag_content, "Missing model_monitoring import"


class TestExcludedPaths:
    """Test that certain paths are excluded from monitoring."""
    
    def test_health_check_excluded(self):
        """Verify health check endpoint is excluded from monitoring."""
        from api.middleware import MonitoringMiddleware
        
        assert '/health' in MonitoringMiddleware.EXCLUDED_PATHS
        assert '/metrics' in MonitoringMiddleware.EXCLUDED_PATHS


class TestDependenciesInstalled:
    """Verify all required dependencies are available."""
    
    def test_scipy_available(self):
        """Verify scipy is installed."""
        try:
            from scipy import stats
            assert stats is not None
        except ImportError:
            pytest.fail("scipy not installed")
    
    def test_numpy_available(self):
        """Verify numpy is installed."""
        try:
            import numpy as np
            assert np is not None
        except ImportError:
            pytest.fail("numpy not installed")
    
    def test_mlflow_available(self):
        """Verify mlflow is installed."""
        try:
            import mlflow
            assert mlflow is not None
        except ImportError:
            pytest.fail("mlflow not installed")
    
    def test_google_cloud_monitoring_available(self):
        """Verify google-cloud-monitoring is installed."""
        try:
            from google.cloud import monitoring_v3
            assert monitoring_v3 is not None
        except ImportError:
            pytest.fail("google-cloud-monitoring not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])