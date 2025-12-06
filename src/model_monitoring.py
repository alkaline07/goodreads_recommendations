"""
Model Monitoring Module for Goodreads Recommendation System

Implements:
1. Performance metric tracking over time
2. Model decay detection with configurable thresholds
3. Data drift detection using statistical tests
4. Integration with Google Cloud Monitoring and MLflow

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import json
import pandas as pd
import numpy as np
from scipy import stats
from google.cloud import bigquery
import mlflow
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MONITORING_DIR = os.path.join(PROJECT_ROOT, "docs", "monitoring_reports")
os.makedirs(MONITORING_DIR, exist_ok=True)

root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(root_env)


class ModelMonitor:
    """
    Comprehensive model monitoring for decay and drift detection.
    """
    
    DECAY_THRESHOLDS = {
        'rmse_increase_pct': 10.0,
        'mae_increase_pct': 10.0,
        'r_squared_decrease': 0.05,
        'accuracy_decrease_pct': 5.0,
    }
    
    DRIFT_THRESHOLDS = {
        'ks_test_pvalue': 0.05,
        'psi_threshold': 0.2,
        'mean_shift_std': 2.0,
    }

    def __init__(self, project_id: Optional[str] = None):
        """Initialize the model monitor."""
        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                airflow_home + "/gcp_credentials.json"
            )
        
        self.bq_client = bigquery.Client(project=project_id)
        self.project_id = self.bq_client.project
        self.dataset_id = "books"
        self.metrics_table = f"{self.project_id}.{self.dataset_id}.model_metrics_history"
        self.drift_table = f"{self.project_id}.{self.dataset_id}.data_drift_history"
        
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("model_monitoring")
        except Exception as e:
            print(f"MLflow initialization warning: {e}")
        
        self._ensure_metrics_tables_exist()
        print(f"ModelMonitor initialized for project: {self.project_id}")

    def _ensure_metrics_tables_exist(self):
        """Create metrics tracking tables if they don't exist."""
        metrics_schema = f"""
        CREATE TABLE IF NOT EXISTS `{self.metrics_table}` (
            timestamp TIMESTAMP,
            model_name STRING,
            model_version STRING,
            metric_name STRING,
            metric_value FLOAT64,
            predictions_count INT64,
            evaluation_dataset STRING
        )
        """
        
        drift_schema = f"""
        CREATE TABLE IF NOT EXISTS `{self.drift_table}` (
            timestamp TIMESTAMP,
            feature_name STRING,
            baseline_mean FLOAT64,
            baseline_std FLOAT64,
            current_mean FLOAT64,
            current_std FLOAT64,
            ks_statistic FLOAT64,
            ks_pvalue FLOAT64,
            psi_score FLOAT64,
            drift_detected BOOL
        )
        """
        
        for schema in [metrics_schema, drift_schema]:
            try:
                self.bq_client.query(schema).result()
            except Exception as e:
                print(f"Table creation warning: {e}")

    def log_metrics(
        self,
        model_name: str,
        predictions_table: str,
        model_version: str = "latest",
        evaluation_dataset: str = "validation"
    ) -> Dict[str, float]:
        """
        Compute and log performance metrics to BigQuery and MLflow.
        """
        print(f"\n[MONITORING] Logging metrics for {model_name} v{model_version}")
        
        metrics = self._compute_metrics(predictions_table)
        if not metrics:
            print("ERROR: Could not compute metrics")
            return {}
        
        timestamp = datetime.utcnow()
        
        self._log_metrics_to_bigquery(
            timestamp, model_name, model_version, metrics, evaluation_dataset
        )
        
        self._log_metrics_to_mlflow(model_name, model_version, metrics)
        
        print(f"  Metrics logged: RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}, R²={metrics['r_squared']:.4f}")
        
        return metrics

    def _compute_metrics(self, predictions_table: str) -> Optional[Dict]:
        """Compute performance metrics from predictions table."""
        query = f"""
        SELECT 
            COUNT(*) as num_predictions,
            AVG(ABS(actual_rating - predicted_rating)) as mae,
            SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) as rmse,
            CORR(actual_rating, predicted_rating) as correlation,
            POWER(CORR(actual_rating, predicted_rating), 2) as r_squared,
            COUNTIF(ABS(actual_rating - predicted_rating) <= 0.5) / COUNT(*) * 100 
                as accuracy_within_0_5,
            COUNTIF(ABS(actual_rating - predicted_rating) <= 1.0) / COUNT(*) * 100 
                as accuracy_within_1_0
        FROM `{predictions_table}`
        WHERE actual_rating IS NOT NULL AND predicted_rating IS NOT NULL
        """
        
        try:
            df = self.bq_client.query(query).to_dataframe(create_bqstorage_client=False)
            if df.empty:
                return None
            
            row = df.iloc[0]
            return {
                'num_predictions': int(row['num_predictions']),
                'mae': float(row['mae']),
                'rmse': float(row['rmse']),
                'r_squared': float(row['r_squared']) if pd.notna(row['r_squared']) else 0.0,
                'correlation': float(row['correlation']) if pd.notna(row['correlation']) else 0.0,
                'accuracy_within_0_5': float(row['accuracy_within_0_5']),
                'accuracy_within_1_0': float(row['accuracy_within_1_0']),
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return None

    def _log_metrics_to_bigquery(
        self,
        timestamp: datetime,
        model_name: str,
        model_version: str,
        metrics: Dict,
        evaluation_dataset: str
    ):
        """Store metrics in BigQuery for historical tracking."""
        rows = []
        for metric_name, metric_value in metrics.items():
            if metric_name == 'num_predictions':
                continue
            rows.append({
                'timestamp': timestamp.isoformat(),
                'model_name': model_name,
                'model_version': model_version,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'predictions_count': metrics['num_predictions'],
                'evaluation_dataset': evaluation_dataset
            })
        
        if rows:
            errors = self.bq_client.insert_rows_json(self.metrics_table, rows)
            if errors:
                print(f"BigQuery insert errors: {errors}")

    def _log_metrics_to_mlflow(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict
    ):
        """Log metrics to MLflow for experiment tracking."""
        try:
            with mlflow.start_run(run_name=f"monitor_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_version", model_version)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        except Exception as e:
            print(f"MLflow logging error: {e}")

    def detect_model_decay(
        self,
        model_name: str,
        lookback_days: int = 30,
        comparison_window_days: int = 7
    ) -> Dict:
        """
        Detect model performance decay by comparing recent metrics to baseline.
        """
        print(f"\n[DECAY DETECTION] Analyzing {model_name}")
        
        baseline_query = f"""
        SELECT 
            metric_name,
            AVG(metric_value) as avg_value,
            STDDEV(metric_value) as std_value
        FROM `{self.metrics_table}`
        WHERE model_name = '{model_name}'
            AND timestamp BETWEEN 
                TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_days} DAY)
                AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {comparison_window_days} DAY)
        GROUP BY metric_name
        """
        
        recent_query = f"""
        SELECT 
            metric_name,
            AVG(metric_value) as avg_value,
            STDDEV(metric_value) as std_value
        FROM `{self.metrics_table}`
        WHERE model_name = '{model_name}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), 
                                           INTERVAL {comparison_window_days} DAY)
        GROUP BY metric_name
        """
        
        try:
            baseline_df = self.bq_client.query(baseline_query).to_dataframe(create_bqstorage_client=False)
            recent_df = self.bq_client.query(recent_query).to_dataframe(create_bqstorage_client=False)
            
            if baseline_df.empty or recent_df.empty:
                print("  Insufficient historical data for decay detection")
                return {'decay_detected': False, 'reason': 'insufficient_data'}
            
            decay_results = {
                'model_name': model_name,
                'timestamp': datetime.utcnow().isoformat(),
                'decay_detected': False,
                'alerts': [],
                'metrics_comparison': {}
            }
            
            baseline_metrics = baseline_df.set_index('metric_name').to_dict('index')
            recent_metrics = recent_df.set_index('metric_name').to_dict('index')
            
            if 'rmse' in baseline_metrics and 'rmse' in recent_metrics:
                baseline_rmse = baseline_metrics['rmse']['avg_value']
                recent_rmse = recent_metrics['rmse']['avg_value']
                rmse_change_pct = ((recent_rmse - baseline_rmse) / baseline_rmse) * 100
                
                decay_results['metrics_comparison']['rmse'] = {
                    'baseline': baseline_rmse,
                    'recent': recent_rmse,
                    'change_pct': rmse_change_pct
                }
                
                if rmse_change_pct > self.DECAY_THRESHOLDS['rmse_increase_pct']:
                    decay_results['decay_detected'] = True
                    decay_results['alerts'].append(
                        f"RMSE increased by {rmse_change_pct:.1f}% "
                        f"(threshold: {self.DECAY_THRESHOLDS['rmse_increase_pct']}%)"
                    )
            
            if 'r_squared' in baseline_metrics and 'r_squared' in recent_metrics:
                baseline_r2 = baseline_metrics['r_squared']['avg_value']
                recent_r2 = recent_metrics['r_squared']['avg_value']
                r2_change = baseline_r2 - recent_r2
                
                decay_results['metrics_comparison']['r_squared'] = {
                    'baseline': baseline_r2,
                    'recent': recent_r2,
                    'change': r2_change
                }
                
                if r2_change > self.DECAY_THRESHOLDS['r_squared_decrease']:
                    decay_results['decay_detected'] = True
                    decay_results['alerts'].append(
                        f"R² decreased by {r2_change:.4f} "
                        f"(threshold: {self.DECAY_THRESHOLDS['r_squared_decrease']})"
                    )
            
            if decay_results['decay_detected']:
                print(f"  WARNING: DECAY DETECTED!")
                for alert in decay_results['alerts']:
                    print(f"    - {alert}")
            else:
                print(f"  OK: No significant decay detected")
            
            self._save_decay_report(decay_results)
            
            return decay_results
            
        except Exception as e:
            print(f"Error in decay detection: {e}")
            return {'decay_detected': False, 'error': str(e)}

    def _save_decay_report(self, results: Dict):
        """Save decay detection report."""
        report_path = os.path.join(
            MONITORING_DIR,
            f"decay_report_{results['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Report saved: {report_path}")

    def detect_data_drift(
        self,
        baseline_table: str,
        current_table: str,
        features: List[str],
        sample_size: int = 10000
    ) -> Dict:
        """
        Detect data drift between baseline and current data distributions.
        """
        print(f"\n[DRIFT DETECTION] Comparing distributions")
        print(f"  Baseline: {baseline_table}")
        print(f"  Current: {current_table}")
        
        drift_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'baseline_table': baseline_table,
            'current_table': current_table,
            'overall_drift_detected': False,
            'features': {}
        }
        
        for feature in features:
            print(f"\n  Analyzing feature: {feature}")
            
            baseline_query = f"""
            SELECT {feature} as value
            FROM `{baseline_table}`
            WHERE {feature} IS NOT NULL
            ORDER BY RAND()
            LIMIT {sample_size}
            """
            
            current_query = f"""
            SELECT {feature} as value
            FROM `{current_table}`
            WHERE {feature} IS NOT NULL
            ORDER BY RAND()
            LIMIT {sample_size}
            """
            
            try:
                baseline_df = self.bq_client.query(baseline_query).to_dataframe(create_bqstorage_client=False)
                current_df = self.bq_client.query(current_query).to_dataframe(create_bqstorage_client=False)
                
                if baseline_df.empty or current_df.empty:
                    print(f"    Skipping {feature}: insufficient data")
                    continue
                
                baseline_values = baseline_df['value'].values.astype(float)
                current_values = current_df['value'].values.astype(float)
                
                ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
                
                psi = self._calculate_psi(baseline_values, current_values)
                
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                current_mean = np.mean(current_values)
                current_std = np.std(current_values)
                
                mean_shift_std = abs(current_mean - baseline_mean) / max(baseline_std, 0.001)
                
                drift_detected = (
                    ks_pvalue < self.DRIFT_THRESHOLDS['ks_test_pvalue'] or
                    psi > self.DRIFT_THRESHOLDS['psi_threshold'] or
                    mean_shift_std > self.DRIFT_THRESHOLDS['mean_shift_std']
                )
                
                feature_result = {
                    'baseline_mean': float(baseline_mean),
                    'baseline_std': float(baseline_std),
                    'current_mean': float(current_mean),
                    'current_std': float(current_std),
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'psi_score': float(psi),
                    'mean_shift_std': float(mean_shift_std),
                    'drift_detected': drift_detected
                }
                
                drift_results['features'][feature] = feature_result
                
                if drift_detected:
                    drift_results['overall_drift_detected'] = True
                    print(f"    WARNING: DRIFT DETECTED - KS p-value: {ks_pvalue:.4f}, "
                          f"PSI: {psi:.4f}, Mean shift: {mean_shift_std:.2f} std")
                else:
                    print(f"    OK: No drift - KS p-value: {ks_pvalue:.4f}, "
                          f"PSI: {psi:.4f}")
                
                self._log_drift_to_bigquery(feature, feature_result)
                
            except Exception as e:
                print(f"    Error analyzing {feature}: {e}")
        
        self._save_drift_report(drift_results)
        
        return drift_results

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift
        """
        _, bin_edges = np.histogram(baseline, bins=bins)
        
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)
        
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)
        
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return psi

    def _log_drift_to_bigquery(self, feature: str, result: Dict):
        """Store drift detection results in BigQuery."""
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'feature_name': feature,
            'baseline_mean': result['baseline_mean'],
            'baseline_std': result['baseline_std'],
            'current_mean': result['current_mean'],
            'current_std': result['current_std'],
            'ks_statistic': result['ks_statistic'],
            'ks_pvalue': result['ks_pvalue'],
            'psi_score': result['psi_score'],
            'drift_detected': result['drift_detected']
        }
        
        try:
            errors = self.bq_client.insert_rows_json(self.drift_table, [row])
            if errors:
                print(f"  BigQuery drift logging error: {errors}")
        except Exception as e:
            print(f"  Error logging drift to BigQuery: {e}")

    def _save_drift_report(self, results: Dict):
        """Save drift detection report."""
        report_path = os.path.join(
            MONITORING_DIR,
            f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Drift report saved: {report_path}")

    def create_alert_if_needed(
        self,
        decay_results: Dict = None,
        drift_results: Dict = None
    ) -> bool:
        """
        Create alerts if issues detected.
        """
        alerts_created = False
        
        if decay_results and decay_results.get('decay_detected'):
            print("\n[ALERT] Model decay detected!")
            self._send_alert_notification(
                alert_type="MODEL_DECAY",
                details=decay_results
            )
            alerts_created = True
        
        if drift_results and drift_results.get('overall_drift_detected'):
            print("\n[ALERT] Data drift detected!")
            self._send_alert_notification(
                alert_type="DATA_DRIFT",
                details=drift_results
            )
            alerts_created = True
        
        return alerts_created

    def _send_alert_notification(self, alert_type: str, details: Dict):
        """Send alert notification."""
        try:
            with mlflow.start_run(run_name=f"alert_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("alert_type", alert_type)
                mlflow.log_param("timestamp", datetime.utcnow().isoformat())
                
                if alert_type == "MODEL_DECAY":
                    for i, alert in enumerate(details.get('alerts', [])):
                        mlflow.log_param(f"alert_message_{i}", alert)
                
                if alert_type == "DATA_DRIFT":
                    drifted_features = [
                        f for f, r in details.get('features', {}).items()
                        if r.get('drift_detected')
                    ]
                    mlflow.log_param("drifted_features", str(drifted_features))
        except Exception as e:
            print(f"  Alert logging error: {e}")


def run_full_monitoring(
    model_name: str = "boosted_tree_regressor",
    predictions_table: Optional[str] = None
) -> Dict:
    """
    Run complete monitoring pipeline.
    """
    monitor = ModelMonitor()
    
    if not predictions_table:
        predictions_table = (
            f"{monitor.project_id}.{monitor.dataset_id}."
            f"{model_name}_rating_predictions"
        )
    
    print("=" * 80)
    print("RUNNING FULL MODEL MONITORING")
    print("=" * 80)
    
    print("\n[STEP 1/3] Logging Current Metrics")
    metrics = monitor.log_metrics(
        model_name=model_name,
        predictions_table=predictions_table
    )
    
    print("\n[STEP 2/3] Checking for Model Decay")
    decay_results = monitor.detect_model_decay(model_name=model_name)
    
    print("\n[STEP 3/3] Checking for Data Drift")
    drift_results = monitor.detect_data_drift(
        baseline_table=f"{monitor.project_id}.{monitor.dataset_id}.goodreads_train_set",
        current_table=f"{monitor.project_id}.{monitor.dataset_id}.goodreads_features",
        features=[
            'avg_rating',
            'ratings_count',
            'book_age_years',
            'text_reviews_count'
        ]
    )
    
    monitor.create_alert_if_needed(decay_results, drift_results)
    
    print("\n" + "=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)
    
    return {
        'metrics': metrics,
        'decay_results': decay_results,
        'drift_results': drift_results
    }


if __name__ == "__main__":
    run_full_monitoring()
