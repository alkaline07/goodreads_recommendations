"""
Model Manager for Vertex AI Model Registry

This module handles intelligent model version management with automatic rollback capability:
1. Compares performance metrics between new and current default model versions
2. Promotes the new version to default if it performs better
3. Keeps the previous version as default if the new model performs worse
4. Logs all decisions and metrics to MLflow for tracking

Performance Comparison Criteria:
- Primary metric: RMSE (lower is better)
- Secondary metrics: MAE, R² (for additional validation)
- Configurable thresholds for minimum improvement required

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import bigquery
import mlflow


def safe_mlflow_log(func, *args, **kwargs):
    """Safely log to MLflow, continue if it fails."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"MLflow logging warning: {e}")
        return None


class ModelManager:
    """
    Manages model version based on performance comparison.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        vertex_region: str = "us-central1",
        improvement_threshold: float = 0.0
    ):
        """
        Initialize the rollback manager.

        Args:
            project_id: GCP project ID
            vertex_region: Vertex AI region
            improvement_threshold: Minimum % improvement required (0.0 = any improvement)
                                   e.g., 0.02 means new model must be 2% better
        """
        airflow_home = os.environ.get("AIRFLOW_HOME", "")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
                airflow_home, "gcp_credentials.json"
            )

        self.bq_client = bigquery.Client(project=project_id)
        self.project_id = self.bq_client.project
        self.vertex_region = vertex_region
        self.improvement_threshold = improvement_threshold
        
        aiplatform.init(project=self.project_id, location=self.vertex_region)

        # Initialize MLflow
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
            mlflow.set_experiment("bigquery_ml_training")
            print("MLflow tracking initialized")
        except Exception as e:
            print(f"MLflow initialization warning: {e}")

        print(f"ModelRollbackManager initialized for project: {self.project_id}")
        print(f"Improvement threshold: {self.improvement_threshold * 100}%")

    def get_model_versions(self, display_name: str) -> List[aiplatform.Model]:
        """
        Get all versions of a model from Vertex AI Model Registry.

        Args:
            display_name: Display name of the model in Vertex AI

        Returns:
            List of Model objects sorted by version (descending)
        """
        try:
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=self.vertex_region,
                order_by="create_time desc"
            )
            
            if not models:
                print(f"No models found with display name: {display_name}")
                return []
            
            print(f"Found {len(models)} version(s) of model '{display_name}'")
            for model in models:
                version_id = model.version_id if hasattr(model, 'version_id') else 'N/A'
                is_default = getattr(model, 'version_aliases', [])
                print(f"  - Version {version_id}, Aliases: {is_default}")
            
            return models

        except Exception as e:
            print(f"Error listing model versions: {e}")
            return []

    def get_model_metrics_from_bq(
        self,
        predictions_table: str
    ) -> Optional[Dict[str, float]]:
        """
        Compute performance metrics from a BigQuery predictions table.

        Args:
            predictions_table: Full table path (project.dataset.table)

        Returns:
            Dictionary with performance metrics or None if error
        """
        query = f"""
        SELECT 
            COUNT(*) as num_predictions,
            AVG(ABS(actual_rating - predicted_rating)) as mae,
            SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) as rmse,
            CORR(actual_rating, predicted_rating) as correlation,
            POWER(CORR(actual_rating, predicted_rating), 2) as r_squared,
            AVG(predicted_rating) as mean_predicted,
            AVG(actual_rating) as mean_actual,
            STDDEV(actual_rating - predicted_rating) as std_error
        FROM `{predictions_table}`
        WHERE actual_rating IS NOT NULL 
            AND predicted_rating IS NOT NULL
        """

        try:
            df = self.bq_client.query(query).to_dataframe(create_bqstorage_client=False)
            if df.empty:
                print(f"No data found in predictions table: {predictions_table}")
                return None

            row = df.iloc[0]
            metrics = {
                'num_predictions': int(row['num_predictions']),
                'mae': float(row['mae']),
                'rmse': float(row['rmse']),
                'r_squared': float(row['r_squared']) if row['r_squared'] is not None else 0.0,
                'correlation': float(row['correlation']) if row['correlation'] is not None else 0.0,
                'mean_predicted': float(row['mean_predicted']),
                'mean_actual': float(row['mean_actual']),
                'std_error': float(row['std_error']) if row['std_error'] is not None else 0.0
            }

            print(f"Metrics from {predictions_table}:")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²: {metrics['r_squared']:.4f}")
            print(f"  Predictions: {metrics['num_predictions']:,}")

            return metrics

        except Exception as e:
            print(f"Error computing metrics from BigQuery: {e}")
            return None

    def compare_models(
        self,
        new_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare new model metrics against current model metrics.

        Args:
            new_metrics: Metrics for the newly trained model
            current_metrics: Metrics for the current default model

        Returns:
            Tuple of (should_promote, improvements_dict)
            - should_promote: True if new model should become default
            - improvements_dict: Dictionary showing metric improvements
        """
        improvements = {}
        
        # Calculate improvements (negative = worse, positive = better)
        # For RMSE and MAE, lower is better, so we invert the calculation
        improvements['rmse_improvement'] = (
            (current_metrics['rmse'] - new_metrics['rmse']) / current_metrics['rmse'] * 100
        )
        improvements['mae_improvement'] = (
            (current_metrics['mae'] - new_metrics['mae']) / current_metrics['mae'] * 100
        )
        
        # For R², higher is better
        improvements['r_squared_improvement'] = (
            (new_metrics['r_squared'] - current_metrics['r_squared']) / 
            max(current_metrics['r_squared'], 0.01) * 100
        )

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Current Model RMSE: {current_metrics['rmse']:.4f}")
        print(f"New Model RMSE:     {new_metrics['rmse']:.4f}")
        print(f"RMSE Improvement:   {improvements['rmse_improvement']:+.2f}%")
        print("-"*60)
        print(f"Current Model MAE:  {current_metrics['mae']:.4f}")
        print(f"New Model MAE:      {new_metrics['mae']:.4f}")
        print(f"MAE Improvement:    {improvements['mae_improvement']:+.2f}%")
        print("-"*60)
        print(f"Current Model R²:   {current_metrics['r_squared']:.4f}")
        print(f"New Model R²:       {new_metrics['r_squared']:.4f}")
        print(f"R² Improvement:     {improvements['r_squared_improvement']:+.2f}%")
        print("="*60)

        # Decision logic: primary metric is RMSE
        rmse_improvement_pct = improvements['rmse_improvement'] / 100
        should_promote = rmse_improvement_pct > self.improvement_threshold

        # Additional validation: ensure other metrics don't drastically degrade
        mae_degradation = improvements['mae_improvement'] < -10  # More than 10% worse
        r2_degradation = improvements['r_squared_improvement'] < -10

        if should_promote and (mae_degradation or r2_degradation):
            print("\nWARNING: RMSE improved but other metrics significantly degraded")
            print("Requiring manual review - not promoting automatically")
            should_promote = False

        return should_promote, improvements

    def find_latest_bqml_model(self, dataset_id: str, model_prefix: str) -> Optional[str]:
        """
        Find the latest BQML model with given prefix.

        Args:
            dataset_id: BigQuery dataset ID
            model_prefix: Model name prefix (e.g., 'boosted_tree_regressor_model_')

        Returns:
            Full model ID (project.dataset.model_name) or None
        """
        try:
            dataset_ref = self.bq_client.dataset(dataset_id, project=self.project_id)
            models = list(self.bq_client.list_models(dataset_ref))
            
            latest_model = None
            latest_time = 0
            
            for model in models:
                if model.model_id.startswith(model_prefix):
                    model_timestamp = model.created.timestamp()
                    if model_timestamp > latest_time:
                        latest_time = model_timestamp
                        latest_model = model
            
            if latest_model:
                full_model_id = f"{self.project_id}.{dataset_id}.{latest_model.model_id}"
                print(f"Found latest BQML model: {full_model_id}")
                return full_model_id
            
            return None
        except Exception as e:
            print(f"Error finding latest BQML model: {e}")
            return None

    def generate_predictions_table(
        self,
        bqml_model_id: str,
        val_table: str,
        output_table: str,
        sample_size: int = 5000
    ) -> bool:
        """
        Generate predictions table from a BQML model.

        Args:
            bqml_model_id: Full BQML model ID (project.dataset.model)
            val_table: Validation table to predict on
            output_table: Output table for predictions
            sample_size: Number of samples to predict on

        Returns:
            True if successful, False otherwise
        """
        print(f"\nGenerating predictions...")
        print(f"  Model: {bqml_model_id}")
        print(f"  Output: {output_table}")
        
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        SELECT 
            user_id_clean,
            book_id,
            rating as actual_rating,
            predicted_rating
        FROM ML.PREDICT(MODEL `{bqml_model_id}`, (
            SELECT * FROM `{val_table}`
            WHERE rating IS NOT NULL
            LIMIT {sample_size}
        ))
        """
        
        try:
            self.bq_client.query(query).result()
            print(f"✓ Predictions generated successfully")
            return True
        except Exception as e:
            print(f"✗ Error generating predictions: {e}")
            return False

    def set_default_version(
        self,
        model: aiplatform.Model,
        display_name: str
    ):
        """
        Set a specific model version as the default.

        Args:
            model: Model object to set as default
            display_name: Display name of the model
        """
        try:
            # Remove 'default' alias from all other versions
            all_versions = self.get_model_versions(display_name)
            for version in all_versions:
                if hasattr(version, 'version_aliases') and 'default' in version.version_aliases:
                    print(f"Removing 'default' alias from version {version.version_id}")
                    version.remove_version_aliases(['default'])

            # Add 'default' alias to the target model
            version_id = model.version_id if hasattr(model, 'version_id') else 'N/A'
            print(f"Setting version {version_id} as default")
            model.add_version_aliases(['default'])
            
            print(f"✓ Successfully set version {version_id} as default")

        except Exception as e:
            print(f"Error setting default version: {e}")
            raise

    def manage_model_rollback(
        self,
        display_name: str,
        new_model_predictions_table: str,
        current_model_predictions_table: Optional[str] = None
    ) -> Dict:
        """
        Main method to manage model rollback based on performance comparison.

        Args:
            display_name: Display name of the model in Vertex AI
            new_model_predictions_table: BQ table with new model predictions
            current_model_predictions_table: BQ table with current model predictions
                                            (if None, will look for existing default)

        Returns:
            Dictionary with rollback decision and details
        """
        print("\n" + "="*80)
        print(f"MODEL ROLLBACK MANAGEMENT: {display_name}")
        print("="*80)

        result = {
            'display_name': display_name,
            'timestamp': datetime.now().isoformat(),
            'decision': None,
            'new_metrics': None,
            'current_metrics': None,
            'improvements': None,
            'promoted_version': None
        }

        # Start MLflow run
        run_name = f"rollback_{display_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            safe_mlflow_log(mlflow.log_param, "display_name", display_name)
            safe_mlflow_log(mlflow.log_param, "new_predictions_table", new_model_predictions_table)

            # Get all model versions
            versions = self.get_model_versions(display_name)
            
            if not versions:
                print("ERROR: No model versions found. Cannot perform rollback management.")
                result['decision'] = 'ERROR_NO_VERSIONS'
                return result

            # Get new model metrics (latest version)
            new_model = versions[0]
            print(f"\nEvaluating NEW model version: {new_model.version_id}")
            new_metrics = self.get_model_metrics_from_bq(new_model_predictions_table)
            
            if not new_metrics:
                print("ERROR: Could not compute metrics for new model")
                result['decision'] = 'ERROR_NEW_METRICS'
                return result

            result['new_metrics'] = new_metrics
            safe_mlflow_log(mlflow.log_metric, "new_model_rmse", new_metrics['rmse'])
            safe_mlflow_log(mlflow.log_metric, "new_model_mae", new_metrics['mae'])
            safe_mlflow_log(mlflow.log_metric, "new_model_r_squared", new_metrics['r_squared'])

            # Find current default version
            current_default = None
            for version in versions[1:]:  # Skip first (newest) version
                if hasattr(version, 'version_aliases') and 'default' in version.version_aliases:
                    current_default = version
                    break

            if not current_default:
                print("\nNo current default version found. Promoting new model as default.")
                self.set_default_version(new_model, display_name)
                result['decision'] = 'PROMOTED_FIRST_DEFAULT'
                result['promoted_version'] = new_model.version_id
                safe_mlflow_log(mlflow.log_param, "decision", result['decision'])
                return result

            # Get current default model metrics
            print(f"\nEvaluating CURRENT default version: {current_default.version_id}")
            
            if current_model_predictions_table:
                current_metrics = self.get_model_metrics_from_bq(current_model_predictions_table)
            else:
                print("WARNING: No predictions table provided for current model")
                print("Cannot compare performance. Keeping current default.")
                result['decision'] = 'KEPT_CURRENT_NO_COMPARISON'
                safe_mlflow_log(mlflow.log_param, "decision", result['decision'])
                return result

            if not current_metrics:
                print("ERROR: Could not compute metrics for current default model")
                result['decision'] = 'ERROR_CURRENT_METRICS'
                return result

            result['current_metrics'] = current_metrics
            safe_mlflow_log(mlflow.log_metric, "current_model_rmse", current_metrics['rmse'])
            safe_mlflow_log(mlflow.log_metric, "current_model_mae", current_metrics['mae'])
            safe_mlflow_log(mlflow.log_metric, "current_model_r_squared", current_metrics['r_squared'])

            # Compare models
            should_promote, improvements = self.compare_models(new_metrics, current_metrics)
            result['improvements'] = improvements

            # Log improvements
            safe_mlflow_log(mlflow.log_metric, "rmse_improvement_pct", improvements['rmse_improvement'])
            safe_mlflow_log(mlflow.log_metric, "mae_improvement_pct", improvements['mae_improvement'])
            safe_mlflow_log(mlflow.log_metric, "r_squared_improvement_pct", improvements['r_squared_improvement'])

            # Make decision
            if should_promote:
                print(f"\n✓ NEW MODEL PERFORMS BETTER - PROMOTING to default")
                self.set_default_version(new_model, display_name)
                result['decision'] = 'PROMOTED_NEW_VERSION'
                result['promoted_version'] = new_model.version_id
            else:
                print(f"\n✗ NEW MODEL DOES NOT MEET IMPROVEMENT THRESHOLD")
                print(f"KEEPING CURRENT DEFAULT VERSION: {current_default.version_id}")
                print("Rollback performed automatically (new version kept but not default)")
                result['decision'] = 'ROLLBACK_KEPT_CURRENT'
                result['promoted_version'] = current_default.version_id

            safe_mlflow_log(mlflow.log_param, "decision", result['decision'])
            safe_mlflow_log(mlflow.log_param, "promoted_version", result['promoted_version'])

            print("\n" + "="*80)
            print(f"ROLLBACK MANAGEMENT COMPLETE: {result['decision']}")
            print("="*80)

        return result


def get_selected_model_from_report() -> Optional[Dict]:
    """
    Read the model selection report to find which model was selected.
    
    Returns:
        Dictionary with model_name and predictions_table, or None if not found
    """
    import json
    
    report_path = "../docs/bias_reports/model_selection_report.json"
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        selected = report.get('selected_model')
        if selected:
            print(f"Found selected model from report: {selected['model_name']}")
            return {
                'model_name': selected['model_name'],
                'predictions_table': selected['predictions_table']
            }
    except FileNotFoundError:
        print(f"Model selection report not found: {report_path}")
        print("Will use default model (boosted_tree_regressor)")
    except Exception as e:
        print(f"Error reading model selection report: {e}")
    
    return None


def get_vertex_ai_model_name(model_name: str) -> str:
    """
    Map model name to Vertex AI registry name.
    
    Args:
        model_name: Model name from selection (e.g., 'boosted_tree_regressor')
    
    Returns:
        Vertex AI display name (e.g., 'goodreads_boosted_tree_regressor')
    """
    mapping = {
        'boosted_tree_regressor': 'goodreads_boosted_tree_regressor',
        'matrix_factorization': 'goodreads_matrix_factorization'
        # 'automl_regressor': 'goodreads_automl_regressor'
    }
    
    return mapping.get(model_name, f'goodreads_{model_name}')


def main():
    """
    Main function to run automatic rollback management.
    
    Uses EXISTING prediction tables that were generated earlier in the pipeline:
    - boosted_tree_rating_predictions OR matrix_factorization_rating_predictions
    - Generated by generate_bias_prediction_tables.py after model training
    - Selected by model_selector.py based on performance + fairness
    
    Expected flow:
    1. Train models (bq_model_training.py)
    2. Register to Vertex AI (register_bqml_models.py)
    3. Generate predictions (generate_bias_prediction_tables.py)
    4. Bias pipeline (bias_pipeline.py)
    5. Model selection (model_selector.py) ← SELECTS WHICH MODEL
    6. Rollback management (THIS SCRIPT) ← USES SELECTED MODEL
    """
    import sys
    import json
    
    # Get project info
    from google.cloud import bigquery

    # Configuration
    improvement_threshold = 0.0  # Require any improvement (adjust as needed)

    manager = ModelManager(
        improvement_threshold=improvement_threshold
    )
    
    client = bigquery.Client()
    project_id = client.project
    dataset_id = "books"
    
    print("\n" + "="*80)
    print("AUTOMATIC MODEL ROLLBACK MANAGER")
    print("="*80)
    print("\nNote: This script uses existing prediction tables from the pipeline")
    print("      (generated by generate_bias_prediction_tables.py)")
    
    # Try to read which model was selected from model selection report
    print("\nChecking which model was selected...")
    selected_info = get_selected_model_from_report()
    
    if selected_info:
        model_name = selected_info['model_name']
        new_predictions_table = selected_info['predictions_table']
        model_display_name = get_vertex_ai_model_name(model_name)
        print(f"✓ Using selected model: {model_name}")
        print(f"  Vertex AI name: {model_display_name}")
        print(f"  Predictions table: {new_predictions_table}")
    else:
        # Fallback to boosted tree if no selection report found
        model_name = "boosted_tree_regressor"
        model_display_name = "goodreads_boosted_tree_regressor"
        new_predictions_table = f"{project_id}.{dataset_id}.boosted_tree_rating_predictions"
        print(f"Using default model: {model_name}")

    print(f"\nProcessing: {model_display_name}")
    print("-"*80)
    
    # Check if predictions table exists
    print("\nVerifying prediction table exists...")
    try:
        table = client.get_table(new_predictions_table)
        print(f"✓ Found predictions table: {new_predictions_table}")
        
        # Get row count
        count_query = f"SELECT COUNT(*) as cnt FROM `{new_predictions_table}`"
        count = client.query(count_query).to_dataframe(create_bqstorage_client=False)['cnt'].iloc[0]
        print(f"  Rows: {count:,}")
    except Exception as e:
        print(f"✗ ERROR: Predictions table not found: {new_predictions_table}")
        print(f"  Error: {e}")
        print("\nPlease run the prediction generation first:")
        print("  python src/generate_bias_prediction_tables.py")
        sys.exit(1)
    
    # Find current default version (if exists)
    print("\nChecking for current default version...")
    versions = manager.get_model_versions(model_display_name)
    current_default = None
    
    for version in versions:
        if hasattr(version, 'version_aliases') and 'default' in version.version_aliases:
            current_default = version
            break
    
    # For rollback comparison, we compare:
    # - NEW model: Latest registered version (versions[0])
    # - CURRENT model: Current default version
    #
    # Both use the SAME predictions table since we evaluate on test set
    # The predictions table reflects the performance of the new model
    # If there's a current default, we need its historical performance
    
    if not current_default:
        print("No current default found.")
        print("This is the first model deployment - will set as default automatically.")
        
        # Set latest version as default since this is first deployment
        new_model = versions[0] if versions else None
        if new_model:
            manager.set_default_version(new_model, model_display_name)
            print(f"\n✓ Set version {new_model.version_id} as default (first deployment)")
            sys.exit(0)
        else:
            print("✗ ERROR: No model versions found in Vertex AI")
            print("Please register model first: python src/register_bqml_models.py")
            sys.exit(1)
    
    # If there's a current default, we need to compare against it
    # For simplicity, we assume the prediction table has been regenerated for the new model
    # and we compare against stored metrics or a baseline
    print(f"Found current default: version {current_default.version_id}")
    print("\nNote: Comparing new model predictions against current default")
    print("      Assuming prediction table represents new model performance")
    
    # Get new model (latest version that's not default)
    new_model = versions[0]
    if new_model.version_id == current_default.version_id:
        print("\nLatest version is already the default - no rollback needed")
        sys.exit(0)
    
    print(f"New model: version {new_model.version_id}")
    
    # For proper comparison, we need separate predictions tables for new and current
    # This is a limitation - we'll need to generate current model predictions
    print("\nGenerating predictions for CURRENT default model...")
    current_predictions_table = f"{project_id}.{dataset_id}.boosted_tree_rating_predictions_current"
    
    artifact_uri = current_default.artifact_uri if hasattr(current_default, 'artifact_uri') else None
    if artifact_uri and artifact_uri.startswith('bq://'):
        current_bqml_model = artifact_uri.replace('bq://', '')
        val_table = f"{project_id}.{dataset_id}.goodreads_test_set"
        
        if manager.generate_predictions_table(
            bqml_model_id=current_bqml_model,
            val_table=val_table,
            output_table=current_predictions_table
        ):
            print("✓ Generated predictions for current default model")
        else:
            print("⚠ WARNING: Could not generate predictions for current model")
            print("Cannot perform comparison - will keep current default")
            sys.exit(1)
    else:
        print("⚠ WARNING: Could not extract BQML model from current default")
        print("Cannot perform comparison - will keep current default")
        sys.exit(1)
    
    # Run rollback management
    print("\nComparing models and making decision...")
    
    result = manager.manage_model_rollback(
        display_name=model_display_name,
        new_model_predictions_table=new_predictions_table,
        current_model_predictions_table=current_predictions_table
    )

    print("\n" + "="*80)
    print("ROLLBACK MANAGEMENT SUMMARY")
    print("="*80)
    print(f"Decision: {result['decision']}")
    if result.get('promoted_version'):
        print(f"Active Version: {result['promoted_version']}")
    
    if result.get('new_metrics'):
        print(f"\nNew Model RMSE: {result['new_metrics']['rmse']:.4f}")
    if result.get('current_metrics'):
        print(f"Current Model RMSE: {result['current_metrics']['rmse']:.4f}")
    
    if result.get('improvements'):
        print(f"\nRMSE Improvement: {result['improvements']['rmse_improvement']:+.2f}%")
    
    print("="*80)
    
    # Exit code based on decision
    if result['decision'] in ['PROMOTED_NEW_VERSION', 'PROMOTED_FIRST_DEFAULT']:
        print("\n✓ New model is now serving in production")
        sys.exit(0)
    elif result['decision'] == 'ROLLBACK_KEPT_CURRENT':
        print("\n⚠ Rollback executed - current model remains in production")
        sys.exit(0)
    else:
        print("\n✗ Error occurred during rollback management")
        sys.exit(1)


if __name__ == "__main__":
    main()
