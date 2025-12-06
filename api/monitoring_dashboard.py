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
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google.cloud import bigquery
import secrets

from .database import get_bq_client

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
    """Fetch model metrics history from BigQuery."""
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
    LIMIT 1500
    """
    
    try:
        df = client.query(query).to_dataframe()
        return df.to_dict('records')
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            print(f"Table not found: model_metrics_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            print(f"Error fetching metrics history: {e}")
        return []


def get_latest_metrics(client: bigquery.Client) -> Dict[str, Dict]:
    """Get the most recent metrics for each model."""
    project_id = client.project
    
    query = f"""
    WITH latest AS (
        SELECT 
            model_name,
            metric_name,
            metric_value,
            timestamp,
            ROW_NUMBER() OVER (PARTITION BY model_name, metric_name ORDER BY timestamp DESC) as rn
        FROM `{project_id}.books.model_metrics_history`
    )
    SELECT model_name, metric_name, metric_value, timestamp
    FROM latest
    WHERE rn = 1
    """
    
    try:
        df = client.query(query).to_dataframe()
        result = {}
        for _, row in df.iterrows():
            model = row['model_name']
            if model not in result:
                result[model] = {'timestamp': str(row['timestamp']), 'metrics': {}}
            result[model]['metrics'][row['metric_name']] = round(row['metric_value'], 4)
        return result
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            print(f"Table not found: model_metrics_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            print(f"Error fetching latest metrics: {e}")
        return {}


def get_drift_history(client: bigquery.Client, days: int = 7) -> List[Dict]:
    """Fetch data drift history from BigQuery."""
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
    """
    
    try:
        df = client.query(query).to_dataframe()
        return df.to_dict('records')
    except Exception as e:
        if '404' in str(e) or 'not found' in str(e).lower():
            print(f"Table not found: data_drift_history. Run 'python scripts/init_monitoring.py' to create tables.")
        else:
            print(f"Error fetching drift history: {e}")
        return []


def get_prediction_stats(client: bigquery.Client) -> Dict:
    """Get prediction statistics from the latest predictions table."""
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
        df = client.query(query).to_dataframe()
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'model_name': row['model_name'],
            'total_predictions': int(row['total_predictions']),
            'avg_predicted': round(float(row['avg_predicted']), 4),
            'avg_actual': round(float(row['avg_actual']), 4),
            'rmse': round(float(row['rmse']), 4),
            'mae': round(float(row['mae']), 4),
            'accuracy_within_0_5': round(float(row['accuracy_0_5']), 2),
            'accuracy_within_1_0': round(float(row['accuracy_1_0']), 2)
        }
    except Exception as e:
        print(f"Error fetching prediction stats: {e}")
        return {}


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Goodreads ML Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                <div class="chart-container" style="display: flex; align-items: center; justify-content: space-around; height: 300px;">
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
    """
    try:
        client = get_bq_client()
        
        current_stats = get_prediction_stats(client)
        latest_metrics = get_latest_metrics(client)
        history = get_metrics_history(client, days=30)
        
        return {
            "current_stats": current_stats,
            "latest_metrics": latest_metrics,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/drift")
async def get_drift_api(username: str = Depends(verify_admin)):
    """
    Get drift detection data for the dashboard.
    """
    try:
        client = get_bq_client()
        drift_history = get_drift_history(client, days=7)
        
        return {
            "drift_history": drift_history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/health")
async def monitoring_health():
    """Health check for monitoring endpoints (no auth required)."""
    return {"status": "healthy", "service": "monitoring"}


@router.get("/api/api-metrics")
async def get_api_metrics_api(username: str = Depends(verify_admin)):
    """Get API performance metrics."""
    try:
        from .middleware import get_metrics_collector
        from .main import get_frontend_metrics_store
        
        collector = get_metrics_collector()
        api_stats = collector.get_all_stats()
        
        frontend_metrics_store = get_frontend_metrics_store()
        frontend_metrics = frontend_metrics_store[-20:] if frontend_metrics_store else []
        
        return {
            "api_stats": api_stats,
            "frontend_metrics": frontend_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
