"""
Monitoring Dashboard for Goodreads Recommendation System

Provides:
1. /report - Admin-only monitoring dashboard with performance metrics
2. /report/api/metrics - JSON API for metrics data
3. /report/api/drift - JSON API for drift detection data

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google.cloud import bigquery
import secrets

from .database import get_bq_client

logger = logging.getLogger(__name__)


class MonitoringBQClientCache:
    """
    Cached BigQuery client singleton for monitoring endpoints.
    Reuses the same client instance to avoid connection overhead.
    """
    _instance: Optional[bigquery.Client] = None
    _lock = Lock()
    _last_refresh: Optional[datetime] = None
    _refresh_interval = timedelta(hours=1)
    
    @classmethod
    def get_client(cls) -> bigquery.Client:
        with cls._lock:
            now = datetime.utcnow()
            should_refresh = (
                cls._instance is None or
                cls._last_refresh is None or
                (now - cls._last_refresh) > cls._refresh_interval
            )
            
            if should_refresh:
                try:
                    cls._instance = get_bq_client()
                    cls._last_refresh = now
                except Exception as e:
                    logger.error(f"Failed to create BQ client: {e}")
                    if cls._instance is None:
                        raise
            
            return cls._instance


class TTLCache:
    """
    Simple in-memory cache with TTL for monitoring query results.
    Thread-safe implementation.
    """
    def __init__(self, default_ttl_seconds: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._default_ttl = default_ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if datetime.utcnow() > entry['expires_at']:
                del self._cache[key]
                return None
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl),
                'cached_at': datetime.utcnow()
            }
    
    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


monitoring_cache = TTLCache(default_ttl_seconds=300)

QUERY_TIMEOUT_SECONDS = 30
QUERY_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="monitoring_bq_")


def get_cached_bq_client() -> bigquery.Client:
    """Get cached BigQuery client for monitoring."""
    return MonitoringBQClientCache.get_client()


def run_query_with_timeout(
    client: bigquery.Client,
    query: str,
    timeout_seconds: int = QUERY_TIMEOUT_SECONDS
) -> Any:
    """
    Run a BigQuery query with timeout protection.
    Returns DataFrame or raises TimeoutError.
    """
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=10 * 1024 * 1024 * 1024  # 10GB limit
    )
    
    query_job = client.query(query, job_config=job_config)
    
    try:
        return query_job.result(timeout=timeout_seconds).to_dataframe()
    except Exception as e:
        if 'timeout' in str(e).lower() or 'deadline' in str(e).lower():
            query_job.cancel()
            raise TimeoutError(f"Query timed out after {timeout_seconds}s")
        raise

router = APIRouter(prefix="/report", tags=["Monitoring"])

security = HTTPBasic()

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials for protected endpoints."""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_metrics_history(client: bigquery.Client, days: int = 30) -> List[Dict]:
    """Fetch model metrics history from BigQuery with caching and timeout."""
    cache_key = f"metrics_history_{days}"
    cached = monitoring_cache.get(cache_key)
    if cached is not None:
        return cached
    
    project_id = client.project
    
    query = f"""
    SELECT 
        DATE(timestamp) as date,
        model_name,
        metric_name,
        AVG(metric_value) as avg_value,
        MIN(metric_value) as min_value,
        MAX(metric_value) as max_value
    FROM `{project_id}.books.model_metrics_history`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
    GROUP BY date, model_name, metric_name
    ORDER BY date DESC, model_name, metric_name
    LIMIT 500
    """
    
    try:
        df = run_query_with_timeout(client, query)
        result = df.to_dict('records')
        monitoring_cache.set(cache_key, result, ttl_seconds=300)
        return result
    except TimeoutError as e:
        logger.error(f"Metrics history query timed out: {e}")
        return []
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            logger.warning(f"Table not found: model_metrics_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            logger.error(f"Error fetching metrics history: {e}")
        return []


def get_latest_metrics(client: bigquery.Client) -> Dict[str, Dict]:
    """Get the most recent metrics for each model with caching and timeout."""
    cache_key = "latest_metrics"
    cached = monitoring_cache.get(cache_key)
    if cached is not None:
        return cached
    
    project_id = client.project
    
    query = f"""
    WITH recent_data AS (
        SELECT 
            model_name,
            metric_name,
            metric_value,
            timestamp
        FROM `{project_id}.books.model_metrics_history`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    ),
    latest AS (
        SELECT 
            model_name,
            metric_name,
            metric_value,
            timestamp,
            ROW_NUMBER() OVER (PARTITION BY model_name, metric_name ORDER BY timestamp DESC) as rn
        FROM recent_data
    )
    SELECT model_name, metric_name, metric_value, timestamp
    FROM latest
    WHERE rn = 1
    LIMIT 100
    """
    
    try:
        df = run_query_with_timeout(client, query)
        result = {}
        for _, row in df.iterrows():
            model = row['model_name']
            if model not in result:
                result[model] = {'timestamp': str(row['timestamp']), 'metrics': {}}
            result[model]['metrics'][row['metric_name']] = round(row['metric_value'], 4)
        monitoring_cache.set(cache_key, result, ttl_seconds=300)
        return result
    except TimeoutError as e:
        logger.error(f"Latest metrics query timed out: {e}")
        return {}
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            logger.warning(f"Table not found: model_metrics_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            logger.error(f"Error fetching latest metrics: {e}")
        return {}


def get_drift_history(client: bigquery.Client, days: int = 7) -> List[Dict]:
    """Fetch data drift history from BigQuery with caching and timeout."""
    cache_key = f"drift_history_{days}"
    cached = monitoring_cache.get(cache_key)
    if cached is not None:
        return cached
    
    project_id = client.project
    
    query = f"""
    SELECT 
        timestamp,
        feature_name,
        baseline_mean,
        current_mean,
        ks_pvalue,
        psi_score,
        drift_detected
    FROM `{project_id}.books.data_drift_history`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
    ORDER BY timestamp DESC
    LIMIT 500
    """
    
    try:
        df = run_query_with_timeout(client, query)
        result = df.to_dict('records')
        monitoring_cache.set(cache_key, result, ttl_seconds=300)
        return result
    except TimeoutError as e:
        logger.error(f"Drift history query timed out: {e}")
        return []
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            logger.warning(f"Table not found: data_drift_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            logger.error(f"Error fetching drift history: {e}")
        return []


def get_prediction_stats(client: bigquery.Client) -> Dict:
    """Get prediction statistics from the latest predictions table with caching and timeout."""
    cache_key = "prediction_stats"
    cached = monitoring_cache.get(cache_key)
    if cached is not None:
        return cached
    
    project_id = client.project
    
    query = f"""
    SELECT 
        'boosted_tree_regressor' as model_name,
        COUNT(*) as total_predictions,
        AVG(predicted_rating) as avg_predicted,
        AVG(actual_rating) as avg_actual,
        SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) as rmse,
        AVG(ABS(actual_rating - predicted_rating)) as mae,
        COUNTIF(ABS(actual_rating - predicted_rating) <= 0.5) / COUNT(*) * 100 as accuracy_0_5,
        COUNTIF(ABS(actual_rating - predicted_rating) <= 1.0) / COUNT(*) * 100 as accuracy_1_0
    FROM `{project_id}.books.boosted_tree_rating_predictions`
    WHERE actual_rating IS NOT NULL
    """
    
    try:
        df = run_query_with_timeout(client, query, timeout_seconds=45)
        if df.empty:
            return {}
        row = df.iloc[0]
        result = {
            'model_name': row['model_name'],
            'total_predictions': int(row['total_predictions']),
            'avg_predicted': round(float(row['avg_predicted']), 4),
            'avg_actual': round(float(row['avg_actual']), 4),
            'rmse': round(float(row['rmse']), 4),
            'mae': round(float(row['mae']), 4),
            'accuracy_within_0_5': round(float(row['accuracy_0_5']), 2),
            'accuracy_within_1_0': round(float(row['accuracy_1_0']), 2)
        }
        monitoring_cache.set(cache_key, result, ttl_seconds=300)
        return result
    except TimeoutError as e:
        logger.error(f"Prediction stats query timed out: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching prediction stats: {e}")
        return {}


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Goodreads ML Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script type="module">
        import {onCLS, onFID, onLCP, onFCP, onTTFB, onINP} from 'https://unpkg.com/web-vitals@3/dist/web-vitals.attribution.js?module';
        
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const vitals = {};
        let vitalsSent = false;
        
        function sendVitals() {
            if (vitalsSent || Object.keys(vitals).length === 0) return;
            vitalsSent = true;
            
            const payload = {
                sessionId: sessionId,
                page: '/report',
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                metrics: {
                    webVitals: vitals,
                    apiCalls: [],
                    errors: []
                }
            };
            
            fetch('/frontend-metrics', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
                keepalive: true
            }).catch(e => console.warn('Failed to send vitals:', e));
        }
        
        onLCP(metric => { vitals.lcp = Math.round(metric.value); });
        onFID(metric => { vitals.fid = Math.round(metric.value); });
        onCLS(metric => { vitals.cls = parseFloat(metric.value.toFixed(4)); });
        onFCP(metric => { vitals.fcp = Math.round(metric.value); });
        onTTFB(metric => { vitals.ttfb = Math.round(metric.value); });
        onINP(metric => { vitals.inp = Math.round(metric.value); });
        
        setTimeout(sendVitals, 5000);
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'hidden') sendVitals();
        });
        window.addEventListener('pagehide', sendVitals);
    </script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
        }
        
        .header .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        
        .status-dot.warning {
            background: #fbbf24;
        }
        
        .status-dot.error {
            background: #f87171;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card .label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
        }
        
        .metric-card .value.good {
            color: #4ade80;
        }
        
        .metric-card .value.warning {
            color: #fbbf24;
        }
        
        .metric-card .value.bad {
            color: #f87171;
        }
        
        .metric-card .change {
            font-size: 0.85rem;
            margin-top: 8px;
        }
        
        .change.positive {
            color: #4ade80;
        }
        
        .change.negative {
            color: #f87171;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chart-card h3 {
            font-size: 1.1rem;
            margin-bottom: 20px;
            color: #fff;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .drift-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .drift-table th,
        .drift-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .drift-table th {
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }
        
        .drift-table tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .badge.ok {
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }
        
        .badge.drift {
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        
        .refresh-btn:hover {
            opacity: 0.9;
        }
        
        .last-updated {
            color: #64748b;
            font-size: 0.85rem;
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #fff;
        }
        
        .alert-banner {
            background: rgba(248, 113, 113, 0.1);
            border: 1px solid rgba(248, 113, 113, 0.3);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display: none;
        }
        
        .alert-banner.show {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .alert-banner .icon {
            font-size: 1.5rem;
        }
        
        .alert-banner .message {
            flex: 1;
        }
        
        .alert-banner .message strong {
            display: block;
            color: #f87171;
            margin-bottom: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Goodreads ML Monitoring Dashboard</h1>
        <div class="status">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">System Healthy</span>
            <span class="last-updated" id="lastUpdated"></span>
        </div>
    </div>
    
    <div class="container">
        <div class="alert-banner" id="alertBanner">
            <span class="icon">⚠️</span>
            <div class="message">
                <strong>Model Performance Alert</strong>
                <span id="alertMessage"></span>
            </div>
        </div>
        
        <div class="alert-banner" id="noDataBanner" style="background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.3);">
            <span class="icon">ℹ️</span>
            <div class="message">
                <strong>No Monitoring Data Available</strong>
                <span>Run the initialization script to create tables and optionally add sample data: <code>python scripts/init_monitoring.py</code></span>
            </div>
        </div>
        
        <h2 class="section-title">API Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Requests</div>
                <div class="value" id="apiTotalRequests">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Error Rate</div>
                <div class="value" id="apiErrorRate">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Latency (P95)</div>
                <div class="value" id="apiP95Latency">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Active Requests</div>
                <div class="value" id="apiActiveRequests">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Requests/Min</div>
                <div class="value" id="apiRpm">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Uptime</div>
                <div class="value" id="apiUptime">--</div>
            </div>
        </div>
        
        <div class="charts-grid" style="margin-bottom: 40px;">
            <div class="chart-card">
                <h3>Top Endpoints by Request Count</h3>
                <div class="chart-container">
                    <canvas id="endpointsChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>Frontend Web Vitals (LCP, FID, CLS)</h3>
                <div class="chart-container" style="display: flex; align-items: center; justify-content: space-around; height: 300px;" id="webVitalsContainer">
                    <div id="webVitalsNoData" style="text-align: center; color: #64748b; display: none;">
                        <div style="font-size: 1.5rem; margin-bottom: 12px;">Collecting Web Vitals...</div>
                        <div style="font-size: 0.9rem; max-width: 400px;">Web Vitals are collected after page interactions. Refresh in a few seconds or interact with the page to trigger LCP/FID/CLS metrics.</div>
                    </div>
                    <div id="webVitalsData" style="display: flex; align-items: center; justify-content: space-around; width: 100%;">
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; font-weight: bold; color: #4ade80;" id="lcpValue">--</div>
                            <div style="color: #94a3b8; margin-top: 8px;">LCP (ms)</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">< 2500 good</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; font-weight: bold; color: #4ade80;" id="fidValue">--</div>
                            <div style="color: #94a3b8; margin-top: 8px;">FID (ms)</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">< 100 good</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; font-weight: bold; color: #4ade80;" id="clsValue">--</div>
                            <div style="color: #94a3b8; margin-top: 8px;">CLS</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">< 0.1 good</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-title">ML Model Performance</h2>
        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <div class="label">RMSE</div>
                <div class="value" id="rmseValue">--</div>
                <div class="change" id="rmseChange"></div>
            </div>
            <div class="metric-card">
                <div class="label">MAE</div>
                <div class="value" id="maeValue">--</div>
                <div class="change" id="maeChange"></div>
            </div>
            <div class="metric-card">
                <div class="label">R² Score</div>
                <div class="value" id="r2Value">--</div>
                <div class="change" id="r2Change"></div>
            </div>
            <div class="metric-card">
                <div class="label">Accuracy (±0.5)</div>
                <div class="value" id="acc05Value">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Accuracy (±1.0)</div>
                <div class="value" id="acc10Value">--</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Predictions</div>
                <div class="value" id="predictionsValue">--</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h3>RMSE & MAE Over Time</h3>
                <div class="chart-container">
                    <canvas id="errorChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>Accuracy Trends</h3>
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
            </div>
        </div>
        
        <h2 class="section-title">Data Drift Detection</h2>
        <div class="chart-card">
            <table class="drift-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Baseline Mean</th>
                        <th>Current Mean</th>
                        <th>KS P-Value</th>
                        <th>PSI Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="driftTableBody">
                    <tr>
                        <td colspan="6" style="text-align: center; color: #64748b;">Loading drift data...</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div style="margin-top: 30px; text-align: center;">
            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
        </div>
    </div>
    
    <script>
        let errorChart, accuracyChart, endpointsChart;
        
        async function fetchMetrics() {
            try {
                const response = await fetch('/report/api/metrics');
                return await response.json();
            } catch (e) {
                console.error('Error fetching metrics:', e);
                return null;
            }
        }
        
        async function fetchDrift() {
            try {
                const response = await fetch('/report/api/drift');
                return await response.json();
            } catch (e) {
                console.error('Error fetching drift:', e);
                return null;
            }
        }
        
        async function fetchApiMetrics() {
            try {
                const response = await fetch('/report/api/api-metrics');
                return await response.json();
            } catch (e) {
                console.error('Error fetching API metrics:', e);
                return null;
            }
        }
        
        function updateMetricCards(data) {
            if (!data || !data.current_stats) return;
            
            const stats = data.current_stats;
            
            document.getElementById('rmseValue').textContent = stats.rmse?.toFixed(4) || '--';
            document.getElementById('maeValue').textContent = stats.mae?.toFixed(4) || '--';
            document.getElementById('acc05Value').textContent = (stats.accuracy_within_0_5?.toFixed(1) || '--') + '%';
            document.getElementById('acc10Value').textContent = (stats.accuracy_within_1_0?.toFixed(1) || '--') + '%';
            document.getElementById('predictionsValue').textContent = stats.total_predictions?.toLocaleString() || '--';
            
            // Color code RMSE
            const rmseEl = document.getElementById('rmseValue');
            if (stats.rmse < 1.0) {
                rmseEl.className = 'value good';
            } else if (stats.rmse < 1.5) {
                rmseEl.className = 'value warning';
            } else {
                rmseEl.className = 'value bad';
            }
            
            // Get R² from latest metrics if available
            if (data.latest_metrics) {
                const model = Object.keys(data.latest_metrics)[0];
                if (model && data.latest_metrics[model].metrics.r_squared !== undefined) {
                    const r2 = data.latest_metrics[model].metrics.r_squared;
                    document.getElementById('r2Value').textContent = r2.toFixed(4);
                    
                    const r2El = document.getElementById('r2Value');
                    if (r2 > 0.5) {
                        r2El.className = 'value good';
                    } else if (r2 > 0.3) {
                        r2El.className = 'value warning';
                    } else {
                        r2El.className = 'value bad';
                    }
                }
            }
        }
        
        function updateCharts(data) {
            if (!data || !data.history || data.history.length === 0) return;
            
            // Process data for charts
            const dates = [...new Set(data.history.map(h => h.date))].sort();
            const rmseData = [];
            const maeData = [];
            const acc05Data = [];
            const acc10Data = [];
            
            dates.forEach(date => {
                const dayMetrics = data.history.filter(h => h.date === date);
                const rmse = dayMetrics.find(m => m.metric_name === 'rmse');
                const mae = dayMetrics.find(m => m.metric_name === 'mae');
                const acc05 = dayMetrics.find(m => m.metric_name === 'accuracy_within_0_5');
                const acc10 = dayMetrics.find(m => m.metric_name === 'accuracy_within_1_0');
                
                rmseData.push(rmse ? rmse.avg_value : null);
                maeData.push(mae ? mae.avg_value : null);
                acc05Data.push(acc05 ? acc05.avg_value : null);
                acc10Data.push(acc10 ? acc10.avg_value : null);
            });
            
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            };
            
            // Error chart
            const errorCtx = document.getElementById('errorChart').getContext('2d');
            if (errorChart) errorChart.destroy();
            errorChart = new Chart(errorCtx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'RMSE',
                            data: rmseData,
                            borderColor: '#f87171',
                            backgroundColor: 'rgba(248, 113, 113, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'MAE',
                            data: maeData,
                            borderColor: '#fbbf24',
                            backgroundColor: 'rgba(251, 191, 36, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: chartOptions
            });
            
            // Accuracy chart
            const accCtx = document.getElementById('accuracyChart').getContext('2d');
            if (accuracyChart) accuracyChart.destroy();
            accuracyChart = new Chart(accCtx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Accuracy ±0.5',
                            data: acc05Data,
                            borderColor: '#4ade80',
                            backgroundColor: 'rgba(74, 222, 128, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Accuracy ±1.0',
                            data: acc10Data,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: chartOptions
            });
        }
        
        function updateDriftTable(data) {
            const tbody = document.getElementById('driftTableBody');
            
            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #64748b;">No drift data available. Run monitoring to collect data.</td></tr>';
                return;
            }
            
            // Group by feature, keep latest
            const latestByFeature = {};
            data.forEach(row => {
                if (!latestByFeature[row.feature_name] || 
                    new Date(row.timestamp) > new Date(latestByFeature[row.feature_name].timestamp)) {
                    latestByFeature[row.feature_name] = row;
                }
            });
            
            let html = '';
            let hasDrift = false;
            
            Object.values(latestByFeature).forEach(row => {
                const isDrift = row.drift_detected;
                if (isDrift) hasDrift = true;
                
                html += `
                    <tr>
                        <td>${row.feature_name}</td>
                        <td>${row.baseline_mean?.toFixed(4) || '--'}</td>
                        <td>${row.current_mean?.toFixed(4) || '--'}</td>
                        <td>${row.ks_pvalue?.toFixed(4) || '--'}</td>
                        <td>${row.psi_score?.toFixed(4) || '--'}</td>
                        <td><span class="badge ${isDrift ? 'drift' : 'ok'}">${isDrift ? 'DRIFT' : 'OK'}</span></td>
                    </tr>
                `;
            });
            
            tbody.innerHTML = html;
            
            // Update status
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const alertBanner = document.getElementById('alertBanner');
            
            if (hasDrift) {
                statusDot.className = 'status-dot warning';
                statusText.textContent = 'Data Drift Detected';
                alertBanner.classList.add('show');
                document.getElementById('alertMessage').textContent = 
                    'Some features show significant distribution shift. Consider retraining the model.';
            }
        }
        
        function updateApiMetrics(data) {
            if (!data || !data.api_stats) return;
            
            const summary = data.api_stats.summary;
            const endpoints = data.api_stats.endpoints;
            const frontendMetrics = data.frontend_metrics || [];
            
            document.getElementById('apiTotalRequests').textContent = summary.total_requests.toLocaleString();
            document.getElementById('apiErrorRate').textContent = summary.error_rate.toFixed(2) + '%';
            document.getElementById('apiP95Latency').textContent = summary.p95_latency_ms.toFixed(0) + ' ms';
            document.getElementById('apiActiveRequests').textContent = summary.active_requests;
            document.getElementById('apiRpm').textContent = summary.requests_per_minute.toFixed(1);
            document.getElementById('apiUptime').textContent = summary.uptime_human;
            
            const errorRateEl = document.getElementById('apiErrorRate');
            if (summary.error_rate < 1) {
                errorRateEl.className = 'value good';
            } else if (summary.error_rate < 5) {
                errorRateEl.className = 'value warning';
            } else {
                errorRateEl.className = 'value bad';
            }
            
            const p95El = document.getElementById('apiP95Latency');
            if (summary.p95_latency_ms < 300) {
                p95El.className = 'value good';
            } else if (summary.p95_latency_ms < 1000) {
                p95El.className = 'value warning';
            } else {
                p95El.className = 'value bad';
            }
            
            if (endpoints && endpoints.length > 0) {
                const topEndpoints = endpoints.slice(0, 5);
                const endpointLabels = topEndpoints.map(e => e.endpoint);
                const endpointCounts = topEndpoints.map(e => e.request_count);
                const endpointLatencies = topEndpoints.map(e => e.avg_latency_ms);
                
                const endpointsCtx = document.getElementById('endpointsChart').getContext('2d');
                if (endpointsChart) endpointsChart.destroy();
                endpointsChart = new Chart(endpointsCtx, {
                    type: 'bar',
                    data: {
                        labels: endpointLabels,
                        datasets: [{
                            label: 'Request Count',
                            data: endpointCounts,
                            backgroundColor: 'rgba(59, 130, 246, 0.5)',
                            borderColor: '#3b82f6',
                            borderWidth: 1,
                            yAxisID: 'y'
                        }, {
                            label: 'Avg Latency (ms)',
                            data: endpointLatencies,
                            backgroundColor: 'rgba(251, 191, 36, 0.5)',
                            borderColor: '#fbbf24',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: '#94a3b8' }
                            }
                        },
                        scales: {
                            x: {
                                ticks: { color: '#64748b' },
                                grid: { color: 'rgba(255,255,255,0.05)' }
                            },
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                ticks: { color: '#64748b' },
                                grid: { color: 'rgba(255,255,255,0.05)' },
                                title: { display: true, text: 'Requests', color: '#94a3b8' }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                ticks: { color: '#64748b' },
                                grid: { drawOnChartArea: false },
                                title: { display: true, text: 'Latency (ms)', color: '#94a3b8' }
                            }
                        }
                    }
                });
            }
            
            const webVitalsNoData = document.getElementById('webVitalsNoData');
            const webVitalsData = document.getElementById('webVitalsData');
            
            if (frontendMetrics.length > 0) {
                const recentMetrics = frontendMetrics.slice(-10);
                let lcpSum = 0, lcpCount = 0;
                let fidSum = 0, fidCount = 0;
                let clsSum = 0, clsCount = 0;
                
                recentMetrics.forEach(entry => {
                    const vitals = entry.metrics?.webVitals || {};
                    if (vitals.lcp) { lcpSum += vitals.lcp; lcpCount++; }
                    if (vitals.fid) { fidSum += vitals.fid; fidCount++; }
                    if (vitals.cls) { clsSum += vitals.cls; clsCount++; }
                });
                
                const hasAnyVitals = lcpCount > 0 || fidCount > 0 || clsCount > 0;
                
                if (hasAnyVitals) {
                    webVitalsNoData.style.display = 'none';
                    webVitalsData.style.display = 'flex';
                    
                    if (lcpCount > 0) {
                        const lcpAvg = lcpSum / lcpCount;
                        const lcpEl = document.getElementById('lcpValue');
                        lcpEl.textContent = lcpAvg.toFixed(0);
                        lcpEl.style.color = lcpAvg < 2500 ? '#4ade80' : lcpAvg < 4000 ? '#fbbf24' : '#f87171';
                    }
                    
                    if (fidCount > 0) {
                        const fidAvg = fidSum / fidCount;
                        const fidEl = document.getElementById('fidValue');
                        fidEl.textContent = fidAvg.toFixed(0);
                        fidEl.style.color = fidAvg < 100 ? '#4ade80' : fidAvg < 300 ? '#fbbf24' : '#f87171';
                    }
                    
                    if (clsCount > 0) {
                        const clsAvg = clsSum / clsCount;
                        const clsEl = document.getElementById('clsValue');
                        clsEl.textContent = clsAvg.toFixed(3);
                        clsEl.style.color = clsAvg < 0.1 ? '#4ade80' : clsAvg < 0.25 ? '#fbbf24' : '#f87171';
                    }
                } else {
                    webVitalsNoData.style.display = 'block';
                    webVitalsData.style.display = 'none';
                }
            } else {
                webVitalsNoData.style.display = 'block';
                webVitalsData.style.display = 'none';
            }
        }
        
        async function refreshData() {
            document.getElementById('lastUpdated').textContent = 'Refreshing...';
            
            const [metricsData, driftData, apiMetrics] = await Promise.all([
                fetchMetrics(),
                fetchDrift(),
                fetchApiMetrics()
            ]);
            
            const hasModelData = metricsData && 
                (metricsData.current_stats && Object.keys(metricsData.current_stats).length > 0 ||
                 metricsData.history && metricsData.history.length > 0 ||
                 metricsData.latest_metrics && Object.keys(metricsData.latest_metrics).length > 0);
            
            const hasDriftData = driftData && driftData.drift_history && driftData.drift_history.length > 0;
            
            const noDataBanner = document.getElementById('noDataBanner');
            if (!hasModelData && !hasDriftData) {
                noDataBanner.classList.add('show');
            } else {
                noDataBanner.style.display = 'none';
            }
            
            if (metricsData) {
                updateMetricCards(metricsData);
                updateCharts(metricsData);
            }
            
            if (driftData) {
                updateDriftTable(driftData.drift_history);
            }
            
            if (apiMetrics) {
                updateApiMetrics(apiMetrics);
            }
            
            document.getElementById('lastUpdated').textContent = 
                'Last updated: ' + new Date().toLocaleTimeString();
        }
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 5 minutes
        setInterval(refreshData, 300000);
    </script>
</body>
</html>
"""


@router.get("", response_class=HTMLResponse)
async def monitoring_dashboard(username: str = Depends(verify_admin)):
    """
    Render the monitoring dashboard HTML page.
    Protected by HTTP Basic Auth - admin only.
    """
    return HTMLResponse(content=DASHBOARD_HTML)


@router.get("/api/metrics")
async def get_metrics_api(username: str = Depends(verify_admin)):
    """
    Get metrics data for the dashboard.
    Returns current stats, latest metrics, and historical data.
    Runs queries in parallel for better performance.
    """
    try:
        client = get_cached_bq_client()
        loop = asyncio.get_event_loop()
        
        current_stats_future = loop.run_in_executor(
            QUERY_EXECUTOR, get_prediction_stats, client
        )
        latest_metrics_future = loop.run_in_executor(
            QUERY_EXECUTOR, get_latest_metrics, client
        )
        history_future = loop.run_in_executor(
            QUERY_EXECUTOR, lambda: get_metrics_history(client, days=30)
        )
        
        current_stats, latest_metrics, history = await asyncio.gather(
            current_stats_future,
            latest_metrics_future,
            history_future,
            return_exceptions=True
        )
        
        if isinstance(current_stats, Exception):
            logger.error(f"Error fetching prediction stats: {current_stats}")
            current_stats = {}
        if isinstance(latest_metrics, Exception):
            logger.error(f"Error fetching latest metrics: {latest_metrics}")
            latest_metrics = {}
        if isinstance(history, Exception):
            logger.error(f"Error fetching history: {history}")
            history = []
        
        return {
            "current_stats": current_stats,
            "latest_metrics": latest_metrics,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/drift")
async def get_drift_api(username: str = Depends(verify_admin)):
    """
    Get drift detection data for the dashboard.
    Uses cached client and query timeout protection.
    """
    try:
        client = get_cached_bq_client()
        loop = asyncio.get_event_loop()
        
        drift_history = await loop.run_in_executor(
            QUERY_EXECUTOR, lambda: get_drift_history(client, days=7)
        )
        
        if isinstance(drift_history, Exception):
            logger.error(f"Error fetching drift history: {drift_history}")
            drift_history = []
        
        return {
            "drift_history": drift_history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Drift API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/health")
async def monitoring_health():
    """Health check for monitoring endpoints (no auth required)."""
    return {"status": "healthy", "service": "monitoring"}


@router.get("/api/cache-status")
async def get_cache_status(username: str = Depends(verify_admin)):
    """Get monitoring cache status for debugging."""
    with monitoring_cache._lock:
        cache_entries = {}
        for key, entry in monitoring_cache._cache.items():
            cache_entries[key] = {
                "cached_at": entry['cached_at'].isoformat(),
                "expires_at": entry['expires_at'].isoformat(),
                "is_expired": datetime.utcnow() > entry['expires_at']
            }
    
    return {
        "cache_entries": cache_entries,
        "bq_client_cached": MonitoringBQClientCache._instance is not None,
        "bq_client_last_refresh": MonitoringBQClientCache._last_refresh.isoformat() if MonitoringBQClientCache._last_refresh else None,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/api/cache-clear")
async def clear_cache(username: str = Depends(verify_admin)):
    """Clear monitoring cache (useful for debugging or forcing fresh data)."""
    monitoring_cache.clear()
    return {
        "status": "cache_cleared",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/api/api-metrics")
async def get_api_metrics_api(username: str = Depends(verify_admin)):
    """Get API performance metrics with timeout protection."""
    try:
        from .middleware import get_metrics_collector
        from .main import get_frontend_metrics_store
        
        collector = get_metrics_collector()
        
        loop = asyncio.get_event_loop()
        try:
            api_stats = await asyncio.wait_for(
                loop.run_in_executor(QUERY_EXECUTOR, collector.get_all_stats),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("API metrics collection timed out")
            api_stats = {"error": "Metrics collection timed out", "summary": {"total_requests": 0, "total_errors": 0, "error_rate": 0, "active_requests": 0, "uptime_seconds": 0, "uptime_human": "--", "requests_per_minute": 0, "avg_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0}, "endpoints": [], "recent_errors": []}
        
        frontend_metrics_store = get_frontend_metrics_store()
        frontend_metrics = frontend_metrics_store[-20:] if frontend_metrics_store else []
        
        return {
            "api_stats": api_stats,
            "frontend_metrics": frontend_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"API metrics error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
