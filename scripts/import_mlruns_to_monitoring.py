"""
Import Real MLflow Data into Monitoring Tables

Reads MLflow experiment data from mlruns folder and imports it into
BigQuery monitoring tables for historical analysis.

Usage:
    python scripts/import_mlruns_to_monitoring.py

Or call programmatically:
    from scripts.import_mlruns_to_monitoring import ensure_mlruns_imported
    ensure_mlruns_imported()

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from google.cloud import bigquery
from dotenv import load_dotenv
from datapipeline.scripts.logger_setup import get_logger

load_dotenv()

logger = get_logger("mlflow-importer")


class MLflowToMonitoringImporter:
    """Import MLflow experiment data into monitoring tables."""

    METRIC_MAPPING = {
        'rmse': 'rmse',
        'bt_rmse': 'rmse',
        'new_model_rmse': 'rmse',
        'current_model_rmse': 'rmse',
        'prediction_rmse': 'rmse',

        'mae': 'mae',
        'bt_mean_absolute_error': 'mae',
        'new_model_mae': 'mae',
        'current_model_mae': 'mae',
        'prediction_mae': 'mae',

        'r_squared': 'r_squared',
        'bt_r2_score': 'r_squared',
        'new_model_r_squared': 'r_squared',
        'current_model_r_squared': 'r_squared',
        'prediction_r_squared': 'r_squared',

        'correlation': 'correlation',
        'prediction_correlation': 'correlation',

        'accuracy_within_0_5_pct': 'accuracy_within_0_5',
        'prediction_accuracy_within_0_5_pct': 'accuracy_within_0_5',

        'accuracy_within_1_0_pct': 'accuracy_within_1_0',
        'prediction_accuracy_within_1_0_pct': 'accuracy_within_1_0',

        'accuracy_within_1_5_pct': 'accuracy_within_1_5',
        'prediction_accuracy_within_1_5_pct': 'accuracy_within_1_5',
    }

    def __init__(self, mlruns_path: str = "mlruns", project_id: str = None):
        """Initialize importer."""
        self.mlruns_path = Path(mlruns_path)

        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    airflow_home + "/gcp_credentials.json"
            )

        self.bq_client = bigquery.Client(project=project_id)
        self.project_id = self.bq_client.project
        self.dataset_id = "books"
        self.metrics_table = f"{self.project_id}.{self.dataset_id}.model_metrics_history"

        logger.info("Importer initialized", project=self.project_id,
                    mlruns_path=str(self.mlruns_path))

    def find_all_runs(self) -> List[Path]:
        """Find all MLflow run directories."""
        runs = []

        if not self.mlruns_path.exists():
            logger.warning("MLflow runs path not found", path=str(self.mlruns_path))
            return runs

        for experiment_dir in self.mlruns_path.iterdir():
            if not experiment_dir.is_dir():
                continue

            if experiment_dir.name in ['.trash', 'models']:
                continue

            for run_dir in experiment_dir.iterdir():
                if run_dir.is_dir() and (run_dir / 'meta.yaml').exists():
                    runs.append(run_dir)

        return runs

    def read_run_metadata(self, run_dir: Path) -> Optional[Dict]:
        """Read run metadata from meta.yaml."""
        meta_file = run_dir / 'meta.yaml'

        try:
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
            return meta
        except Exception as e:
            logger.error("Error reading metadata", file=str(meta_file), error=str(e))
            return None

    def read_metric_value(self, metric_file: Path) -> Optional[float]:
        """Read metric value from MLflow metric file."""
        try:
            with open(metric_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
        except Exception as e:
            logger.error("Error reading metric", metric=metric_file.name, error=str(e))
        return None

    def extract_run_metrics(self, run_dir: Path, meta: Dict) -> List[Dict]:
        """Extract all metrics from a run."""
        metrics_dir = run_dir / 'metrics'

        if not metrics_dir.exists():
            return []

        timestamp_ms = meta.get('start_time') or meta.get('end_time')
        if not timestamp_ms:
            logger.warning("No timestamp for run", run=run_dir.name)
            return []

        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
        run_name = meta.get('run_name', run_dir.name)

        records = []

        for metric_file in metrics_dir.iterdir():
            if not metric_file.is_file():
                continue

            metric_name = metric_file.name

            if metric_name in self.METRIC_MAPPING:
                normalized_name = self.METRIC_MAPPING[metric_name]
                value = self.read_metric_value(metric_file)

                if value is not None:
                    records.append({
                        'timestamp': timestamp.isoformat(),
                        'model_name': 'boosted_tree_regressor',
                        'model_version': run_name,
                        'metric_name': normalized_name,
                        'metric_value': value,
                        'predictions_count': 10000,
                        'evaluation_dataset': 'validation'
                    })

        return records

    def import_all_runs(self, dry_run: bool = False) -> int:
        """Import all MLflow runs into monitoring tables."""
        logger.info("Starting MLflow data import", dry_run=dry_run)

        runs = self.find_all_runs()

        if not runs:
            logger.warning("No MLflow runs found")
            return 0

        logger.info("MLflow runs found", count=len(runs))

        all_records = []

        for i, run_dir in enumerate(runs, 1):
            logger.info("Processing run", run=run_dir.name, progress=f"{i}/{len(runs)}")

            meta = self.read_run_metadata(run_dir)
            if not meta:
                continue

            records = self.extract_run_metrics(run_dir, meta)

            if records:
                logger.info("Metrics extracted", run=run_dir.name, count=len(records))
                all_records.extend(records)
            else:
                logger.warning("No relevant metrics found", run=run_dir.name)

        logger.info("Import summary", total_runs=len(runs), total_metrics=len(all_records))

        if all_records:
            metrics_by_type = {}
            for record in all_records:
                metric = record['metric_name']
                metrics_by_type[metric] = metrics_by_type.get(metric, 0) + 1

            logger.info("Metrics breakdown", metrics=metrics_by_type)

        if dry_run:
            logger.info("Dry run completed", sample_records=all_records[:3])
            return len(all_records)

        if all_records:
            logger.info("Inserting records into BigQuery", count=len(all_records),
                        table=self.metrics_table)

            try:
                errors = self.bq_client.insert_rows_json(self.metrics_table, all_records)

                if errors:
                    logger.error("Errors during insertion", errors=errors[:5])
                    return 0
                else:
                    logger.info("Successfully inserted records", count=len(all_records))

                    logger.info("Verifying inserted data")
                    verify_query = f"""
                    SELECT 
                        metric_name,
                        COUNT(*) as count,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM `{self.metrics_table}`
                    GROUP BY metric_name
                    ORDER BY metric_name
                    """

                    df = self.bq_client.query(verify_query).to_dataframe(
                        create_bqstorage_client=False)
                    logger.info("Data verification complete", metrics_count=len(df))

                    return len(all_records)

            except Exception as e:
                logger.error("Error inserting data", error=str(e))
                return 0

        return 0


def has_mlruns_data(mlruns_path: str = "mlruns") -> bool:
    """
    Check if mlruns folder exists and has experiment data.
    
    Returns:
        bool: True if mlruns folder has importable data
    """
    mlruns_dir = Path(mlruns_path)
    
    if not mlruns_dir.exists():
        return False
    
    for experiment_dir in mlruns_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        if experiment_dir.name in ['.trash', 'models']:
            continue
        for run_dir in experiment_dir.iterdir():
            if run_dir.is_dir() and (run_dir / 'meta.yaml').exists():
                return True
    
    return False


def is_monitoring_data_imported(project_id: str = None) -> bool:
    """
    Check if monitoring table already has data.
    
    Returns:
        bool: True if model_metrics_history table has records
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
        
        query = f"""
        SELECT COUNT(*) as count
        FROM `{project_id}.{dataset_id}.model_metrics_history`
        LIMIT 1
        """
        
        result = client.query(query).result()
        for row in result:
            return row.count > 0
        
        return False
        
    except Exception as e:
        logger.warning("Error checking monitoring data", error=str(e))
        return False


def ensure_mlruns_imported(
    mlruns_path: str = "mlruns",
    project_id: str = None,
    silent: bool = True
) -> bool:
    """
    Ensure MLflow data is imported to monitoring tables if available.
    
    This function is idempotent - safe to call multiple times.
    Only imports if:
    1. mlruns folder exists and has data
    2. Monitoring table is empty (no existing data)
    
    Args:
        mlruns_path: Path to mlruns folder
        project_id: GCP project ID (uses default if not provided)
        silent: If True, suppresses most output (for automatic startup)
    
    Returns:
        bool: True if import completed or not needed, False on error
    """
    if not has_mlruns_data(mlruns_path):
        logger.debug("No MLflow data to import")
        return True
    
    if is_monitoring_data_imported(project_id):
        logger.info("Monitoring data already exists - skipping MLflow import")
        if not silent:
            print("Monitoring data already exists - skipping MLflow import")
        return True
    
    logger.info("Importing MLflow data to monitoring tables...")
    if not silent:
        print("Importing MLflow data to monitoring tables...")
    
    try:
        importer = MLflowToMonitoringImporter(mlruns_path=mlruns_path, project_id=project_id)
        count = importer.import_all_runs(dry_run=False)
        
        if count > 0:
            logger.info("MLflow import completed", records=count)
            if not silent:
                print(f"  ✅ Imported {count} metrics from MLflow")
            return True
        else:
            logger.warning("No metrics imported from MLflow")
            return True
            
    except Exception as e:
        logger.error("Error during MLflow import", error=str(e))
        if not silent:
            print(f"  ❌ Error importing MLflow data: {e}")
        return False


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("MLFLOW TO MONITORING DATA IMPORTER")
    print("=" * 80)

    print("\nThis script will:")
    print("  1. Scan mlruns/ folder for experiment data")
    print("  2. Extract metrics (RMSE, MAE, R², accuracy)")
    print("  3. Insert into BigQuery model_metrics_history table")

    print("\n" + "-" * 80)

    mlruns_path = os.environ.get("MLFLOW_RUNS_PATH", "mlruns")

    if not Path(mlruns_path).exists():
        print(f"\n❌ ERROR: mlruns folder not found at: {mlruns_path}")
        print("\nPlease ensure:")
        print("  1. You're in the project root directory")
        print("  2. MLflow experiments have been run (mlruns/ folder exists)")
        print("  3. Or set MLFLOW_RUNS_PATH environment variable")
        sys.exit(1)

    importer = MLflowToMonitoringImporter(mlruns_path=mlruns_path)

    print("\nOptions:")
    print("  1. Dry run (preview data without inserting)")
    print("  2. Import data to BigQuery")
    print("  3. Cancel")

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == '1':
        print("\n[DRY RUN MODE]")
        count = importer.import_all_runs(dry_run=True)
        print(f"\n✅ Dry run complete. Found {count} metrics that would be imported.")

    elif choice == '2':
        print("\n[IMPORT MODE]")
        confirm = input(
            "Are you sure you want to import data to BigQuery? (yes/no): ").strip().lower()

        if confirm == 'yes':
            count = importer.import_all_runs(dry_run=False)

            if count > 0:
                print("\n" + "=" * 80)
                print("✅ IMPORT COMPLETE")
                print("=" * 80)
                print(f"\nImported {count} metrics from MLflow")
                print("\nNext steps:")
                print("  1. Access dashboard: http://localhost:8080/report")
                print("  2. View metrics: curl http://localhost:8080/metrics")
                print(
                    "  3. Run monitoring: python -c 'from src.model_monitoring import run_full_monitoring; run_full_monitoring()'")
            else:
                print("\n❌ Import failed or no data to import")
        else:
            print("Import cancelled")
    else:
        print("Cancelled")


if __name__ == "__main__":
    main()
