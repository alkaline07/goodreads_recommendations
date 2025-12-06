"""
API Monitoring Middleware for Goodreads Recommendation API

Tracks:
1. Request latency (response time)
2. Request counts by endpoint
3. Error rates and status codes
4. Slow request detection
5. Active requests gauge

Stores metrics in-memory with BigQuery persistence for historical analysis.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
import statistics

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint."""
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_request_time: Optional[datetime] = None
    slow_requests: int = 0


class APIMetricsCollector:
    """
    Collects and stores API performance metrics.
    Thread-safe singleton for use across the application.
    """
    
    _instance = None
    _lock = Lock()
    
    SLOW_REQUEST_THRESHOLD_MS = 1000
    MAX_LATENCY_SAMPLES = 1000
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.endpoints: Dict[str, EndpointMetrics] = defaultdict(EndpointMetrics)
        self.global_metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'start_time': datetime.utcnow(),
            'active_requests': 0
        }
        self.recent_errors: List[Dict] = []
        self.request_timeline: List[Dict] = []
        self._lock = Lock()
        self._initialized = True
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        error_message: Optional[str] = None
    ):
        """Record metrics for a single request."""
        with self._lock:
            key = f"{method} {endpoint}"
            metrics = self.endpoints[key]
            
            metrics.request_count += 1
            metrics.total_latency_ms += latency_ms
            metrics.status_codes[status_code] += 1
            metrics.last_request_time = datetime.utcnow()
            
            if len(metrics.latencies) < self.MAX_LATENCY_SAMPLES:
                metrics.latencies.append(latency_ms)
            else:
                metrics.latencies.pop(0)
                metrics.latencies.append(latency_ms)
            
            if latency_ms > self.SLOW_REQUEST_THRESHOLD_MS:
                metrics.slow_requests += 1
            
            self.global_metrics['total_requests'] += 1
            
            is_error = status_code >= 400
            if is_error:
                metrics.error_count += 1
                self.global_metrics['total_errors'] += 1
                
                self.recent_errors.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'endpoint': key,
                    'status_code': status_code,
                    'latency_ms': latency_ms,
                    'error_message': error_message
                })
                if len(self.recent_errors) > 100:
                    self.recent_errors.pop(0)
            
            self.request_timeline.append({
                'timestamp': datetime.utcnow().isoformat(),
                'endpoint': key,
                'status_code': status_code,
                'latency_ms': latency_ms
            })
            if len(self.request_timeline) > 1000:
                self.request_timeline.pop(0)
    
    def increment_active_requests(self):
        """Increment active request counter."""
        with self._lock:
            self.global_metrics['active_requests'] += 1
    
    def decrement_active_requests(self):
        """Decrement active request counter."""
        with self._lock:
            self.global_metrics['active_requests'] = max(
                0, self.global_metrics['active_requests'] - 1
            )
    
    def get_endpoint_stats(self, endpoint: str) -> Dict:
        """Get statistics for a specific endpoint."""
        with self._lock:
            metrics = self.endpoints.get(endpoint)
            if not metrics or metrics.request_count == 0:
                return {}
            
            latencies = metrics.latencies if metrics.latencies else [0]
            
            return {
                'endpoint': endpoint,
                'request_count': metrics.request_count,
                'error_count': metrics.error_count,
                'error_rate': (metrics.error_count / metrics.request_count) * 100,
                'avg_latency_ms': metrics.total_latency_ms / metrics.request_count,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'p50_latency_ms': statistics.median(latencies),
                'p95_latency_ms': self._percentile(latencies, 95),
                'p99_latency_ms': self._percentile(latencies, 99),
                'slow_requests': metrics.slow_requests,
                'status_codes': dict(metrics.status_codes),
                'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }
    
    def get_all_stats(self) -> Dict:
        """Get all API monitoring statistics."""
        with self._lock:
            uptime = datetime.utcnow() - self.global_metrics['start_time']
            total_requests = self.global_metrics['total_requests']
            total_errors = self.global_metrics['total_errors']
            
            all_latencies = []
            for metrics in self.endpoints.values():
                all_latencies.extend(metrics.latencies)
            
            endpoints_stats = []
            for endpoint in self.endpoints:
                stats = self.get_endpoint_stats(endpoint)
                if stats:
                    endpoints_stats.append(stats)
            
            endpoints_stats.sort(key=lambda x: x['request_count'], reverse=True)
            
            return {
                'summary': {
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'error_rate': (total_errors / total_requests * 100) if total_requests > 0 else 0,
                    'active_requests': self.global_metrics['active_requests'],
                    'uptime_seconds': uptime.total_seconds(),
                    'uptime_human': str(uptime).split('.')[0],
                    'requests_per_minute': (total_requests / uptime.total_seconds() * 60) if uptime.total_seconds() > 0 else 0,
                    'avg_latency_ms': (sum(all_latencies) / len(all_latencies)) if all_latencies else 0,
                    'p95_latency_ms': self._percentile(all_latencies, 95) if all_latencies else 0,
                    'p99_latency_ms': self._percentile(all_latencies, 99) if all_latencies else 0,
                },
                'endpoints': endpoints_stats,
                'recent_errors': self.recent_errors[-10:],
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_timeline_data(self, minutes: int = 5) -> Dict:
        """Get request timeline data for charts."""
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
            
            recent = [
                r for r in self.request_timeline
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
            
            by_minute = defaultdict(lambda: {'count': 0, 'errors': 0, 'latencies': []})
            for req in recent:
                minute = req['timestamp'][:16]
                by_minute[minute]['count'] += 1
                by_minute[minute]['latencies'].append(req['latency_ms'])
                if req['status_code'] >= 400:
                    by_minute[minute]['errors'] += 1
            
            timeline = []
            for minute, data in sorted(by_minute.items()):
                timeline.append({
                    'time': minute,
                    'requests': data['count'],
                    'errors': data['errors'],
                    'avg_latency': sum(data['latencies']) / len(data['latencies']) if data['latencies'] else 0
                })
            
            return {
                'timeline': timeline,
                'period_minutes': minutes
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.endpoints.clear()
            self.global_metrics = {
                'total_requests': 0,
                'total_errors': 0,
                'start_time': datetime.utcnow(),
                'active_requests': 0
            }
            self.recent_errors.clear()
            self.request_timeline.clear()
    
    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


metrics_collector = APIMetricsCollector()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for API monitoring.
    Tracks request latency, errors, and throughput.
    """
    
    EXCLUDED_PATHS = {'/health', '/metrics', '/report/api/health'}
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        self.collector.increment_active_requests()
        start_time = time.perf_counter()
        error_message = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error_message = str(e)
            status_code = 500
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            endpoint = self._normalize_path(request.url.path)
            
            self.collector.record_request(
                endpoint=endpoint,
                method=request.method,
                status_code=status_code,
                latency_ms=latency_ms,
                error_message=error_message
            )
            
            self.collector.decrement_active_requests()
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path to group similar endpoints."""
        parts = path.strip('/').split('/')
        normalized = []
        
        for part in parts:
            if part.isdigit() or len(part) > 20:
                normalized.append('{id}')
            else:
                normalized.append(part)
        
        return '/' + '/'.join(normalized) if normalized else '/'


def get_metrics_collector() -> APIMetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector
