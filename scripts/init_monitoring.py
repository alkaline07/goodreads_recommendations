"""
Bootstrap Script for Monitoring System

Creates required BigQuery tables and initializes monitoring infrastructure.

Run this script once before using the monitoring dashboard:
    python scripts/init_monitoring.py

Or call programmatically:
    from scripts.init_monitoring import ensure_monitoring_setup
    ensure_monitoring_setup()

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def initialize_monitoring_tables(project_id: str = None):
    """
    Create monitoring tables in BigQuery if they don't exist.
    """
    print("=" * 80)
    print("INITIALIZING MONITORING INFRASTRUCTURE")
    print("=" * 80)
    
    airflow_home = os.environ.get("AIRFLOW_HOME")
    if airflow_home:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            airflow_home + "/gcp_credentials.json"
        )
    
    client = bigquery.Client(project=project_id)
    project_id = client.project
    dataset_id = "books"
    
    print(f"\nProject ID: {project_id}")
    print(f"Dataset ID: {dataset_id}")
    
    tables_to_create = {
        "model_metrics_history": f"""
        CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.model_metrics_history` (
            timestamp TIMESTAMP,
            model_name STRING,
            model_version STRING,
            metric_name STRING,
            metric_value FLOAT64,
            predictions_count INT64,
            evaluation_dataset STRING
        )
        PARTITION BY DATE(timestamp)
        OPTIONS(
            description="Model performance metrics history for monitoring and decay detection"
        )
        """,
        
        "data_drift_history": f"""
        CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.data_drift_history` (
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
        PARTITION BY DATE(timestamp)
        OPTIONS(
            description="Data drift detection history for monitoring distribution shifts"
        )
        """
    }
    
    print("\n" + "-" * 80)
    print("Creating BigQuery Tables")
    print("-" * 80)
    
    for table_name, create_statement in tables_to_create.items():
        print(f"\nCreating table: {table_name}")
        try:
            query_job = client.query(create_statement)
            query_job.result()
            print(f"  ✅ Table created/verified: {project_id}.{dataset_id}.{table_name}")
        except Exception as e:
            print(f"  ❌ Error creating table {table_name}: {e}")
            return False
    
    print("\n" + "-" * 80)
    print("Verifying Tables Exist")
    print("-" * 80)
    
    for table_name in tables_to_create.keys():
        full_table_id = f"{project_id}.{dataset_id}.{table_name}"
        try:
            table = client.get_table(full_table_id)
            print(f"  ✅ {table_name}: {table.num_rows} rows, {table.num_bytes:,} bytes")
        except Exception as e:
            print(f"  ❌ {table_name}: Not found - {e}")
            return False
    
    print("\n" + "-" * 80)
    print("Inserting Sample Metrics (Optional)")
    print("-" * 80)
    
    print("\nWould you like to insert sample metrics for testing? (y/n)")
    user_input = input().strip().lower()
    
    if user_input == 'y':
        insert_sample_data(client, project_id, dataset_id)
    
    print("\n" + "=" * 80)
    print("MONITORING INITIALIZATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run model monitoring: python -c 'from src.model_monitoring import run_full_monitoring; run_full_monitoring()'")
    print("  2. Access dashboard: http://localhost:8080/report")
    print("  3. View API metrics: http://localhost:8080/metrics")
    print()
    
    return True


def insert_sample_data(client: bigquery.Client, project_id: str, dataset_id: str):
    """Insert sample data for testing the dashboard."""
    from datetime import datetime, timedelta
    import random
    
    print("\nInserting sample metrics data...")
    
    metrics_table = f"{project_id}.{dataset_id}.model_metrics_history"
    drift_table = f"{project_id}.{dataset_id}.data_drift_history"
    
    sample_metrics = []
    for i in range(30):
        timestamp = (datetime.utcnow() - timedelta(days=30-i)).isoformat()
        base_rmse = 0.85
        base_mae = 0.65
        base_r2 = 0.45
        
        rmse = base_rmse + random.uniform(-0.05, 0.1)
        mae = base_mae + random.uniform(-0.03, 0.08)
        r2 = base_r2 + random.uniform(-0.05, 0.05)
        
        sample_metrics.extend([
            {
                'timestamp': timestamp,
                'model_name': 'boosted_tree_regressor',
                'model_version': 'v1.0',
                'metric_name': 'rmse',
                'metric_value': rmse,
                'predictions_count': random.randint(8000, 12000),
                'evaluation_dataset': 'validation'
            },
            {
                'timestamp': timestamp,
                'model_name': 'boosted_tree_regressor',
                'model_version': 'v1.0',
                'metric_name': 'mae',
                'metric_value': mae,
                'predictions_count': random.randint(8000, 12000),
                'evaluation_dataset': 'validation'
            },
            {
                'timestamp': timestamp,
                'model_name': 'boosted_tree_regressor',
                'model_version': 'v1.0',
                'metric_name': 'r_squared',
                'metric_value': r2,
                'predictions_count': random.randint(8000, 12000),
                'evaluation_dataset': 'validation'
            },
            {
                'timestamp': timestamp,
                'model_name': 'boosted_tree_regressor',
                'model_version': 'v1.0',
                'metric_name': 'accuracy_within_0_5',
                'metric_value': random.uniform(40, 50),
                'predictions_count': random.randint(8000, 12000),
                'evaluation_dataset': 'validation'
            }
        ])
    
    errors = client.insert_rows_json(metrics_table, sample_metrics)
    if errors:
        print(f"  ❌ Error inserting metrics: {errors}")
    else:
        print(f"  ✅ Inserted {len(sample_metrics)} sample metric records")
    
    sample_drift = []
    for i in range(7):
        timestamp = (datetime.utcnow() - timedelta(days=7-i)).isoformat()
        
        for feature in ['avg_rating', 'ratings_count', 'book_age_years', 'text_reviews_count']:
            drift_detected = random.random() > 0.8
            
            sample_drift.append({
                'timestamp': timestamp,
                'feature_name': feature,
                'baseline_mean': random.uniform(3.0, 4.5),
                'baseline_std': random.uniform(0.5, 1.5),
                'current_mean': random.uniform(3.0, 4.5),
                'current_std': random.uniform(0.5, 1.5),
                'ks_statistic': random.uniform(0.01, 0.2),
                'ks_pvalue': random.uniform(0.01, 0.5),
                'psi_score': random.uniform(0.05, 0.3),
                'drift_detected': drift_detected
            })
    
    errors = client.insert_rows_json(drift_table, sample_drift)
    if errors:
        print(f"  ❌ Error inserting drift data: {errors}")
    else:
        print(f"  ✅ Inserted {len(sample_drift)} sample drift records")


def is_monitoring_setup_complete(project_id: str = None) -> bool:
    """
    Check if monitoring tables already exist in BigQuery.
    
    Returns:
        bool: True if both monitoring tables exist, False otherwise
    """
    try:
        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                airflow_home + "/gcp_credentials.json"
            )
        
        client = bigquery.Client(project=project_id)
        project_id = client.project
        dataset_id = "books"
        
        required_tables = ["model_metrics_history", "data_drift_history"]
        
        for table_name in required_tables:
            full_table_id = f"{project_id}.{dataset_id}.{table_name}"
            try:
                client.get_table(full_table_id)
            except Exception:
                logger.debug(f"Table not found: {full_table_id}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error checking monitoring setup: {e}")
        return False


def ensure_monitoring_setup(project_id: str = None, silent: bool = True) -> bool:
    """
    Ensure monitoring tables are set up. Creates them if they don't exist.
    
    This function is idempotent - safe to call multiple times.
    
    Args:
        project_id: GCP project ID (uses default if not provided)
        silent: If True, suppresses most output (for automatic startup)
    
    Returns:
        bool: True if setup is complete (either already existed or created successfully)
    """
    if is_monitoring_setup_complete(project_id):
        if not silent:
            print("Monitoring tables already exist - setup complete")
        logger.info("Monitoring setup already complete")
        return True
    
    logger.info("Monitoring tables not found - initializing...")
    if not silent:
        print("Monitoring tables not found - initializing...")
    
    try:
        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                airflow_home + "/gcp_credentials.json"
            )
        
        client = bigquery.Client(project=project_id)
        project_id = client.project
        dataset_id = "books"
        
        tables_to_create = {
            "model_metrics_history": f"""
            CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.model_metrics_history` (
                timestamp TIMESTAMP,
                model_name STRING,
                model_version STRING,
                metric_name STRING,
                metric_value FLOAT64,
                predictions_count INT64,
                evaluation_dataset STRING
            )
            PARTITION BY DATE(timestamp)
            OPTIONS(
                description="Model performance metrics history for monitoring and decay detection"
            )
            """,
            
            "data_drift_history": f"""
            CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.data_drift_history` (
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
            PARTITION BY DATE(timestamp)
            OPTIONS(
                description="Data drift detection history for monitoring distribution shifts"
            )
            """
        }
        
        for table_name, create_statement in tables_to_create.items():
            try:
                query_job = client.query(create_statement)
                query_job.result()
                logger.info(f"Created/verified table: {project_id}.{dataset_id}.{table_name}")
                if not silent:
                    print(f"  ✅ Created table: {table_name}")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                if not silent:
                    print(f"  ❌ Error creating table {table_name}: {e}")
                return False
        
        logger.info("Monitoring setup completed successfully")
        if not silent:
            print("✅ Monitoring setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during monitoring setup: {e}")
        if not silent:
            print(f"❌ Error during monitoring setup: {e}")
        return False


if __name__ == "__main__":
    success = initialize_monitoring_tables()
    sys.exit(0 if success else 1)
