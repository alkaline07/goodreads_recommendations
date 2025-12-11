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
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "bias_reports")
MODEL_SELECTION_DIR = os.path.join(DOCS_DIR, "model_selection")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MODEL_SELECTION_DIR, exist_ok=True)
root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(root_env)
print("Loaded .env from:", root_env)

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
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")

        # Initialize MLflow
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("bigquery_ml_training")
            print("MLflow tracking initialized")
        except Exception as e:
            print(f"MLflow initialization warning: {e}")

        print(f"ModelRollbackManager initialized for project: {self.project_id}")
        print(f"Improvement threshold: {self.improvement_threshold * 100}%")

    def get_model_versions(self, display_name: str) -> List:
        """
        Get all versions of a model from Vertex AI Model Registry.
        Returns a list of VersionInfo objects sorted by version ID (descending).
        """
        try:
            # 1. Find the Parent Model
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=self.vertex_region
            )
            
            if not models:
                print(f"No models found with display name: {display_name}")
                return []
            
            parent_model = models[0]
            
            # 2. Get version history
            # These are VersionInfo objects, NOT full Model resources
            versions = parent_model.versioning_registry.list_versions()
            
            print(f"Found {len(versions)} version(s) of model '{display_name}'")
            
            # Debug print to help inspect attributes if needed
            for v in versions:
                print(f"ID: {v.version_id}, Aliases: {v.version_aliases}")

            # 3. Sort by version_id (converted to int) to get the latest
            versions.sort(key=lambda x: int(x.version_id), reverse=True)
            
            return versions

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
            model: The Model object representing the version to set as default
            display_name: Display name of the model in Vertex AI
        """
        try:
            # Get the parent model to use with ModelRegistry
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=self.vertex_region
            )
            if not models:
                raise ValueError(f"No models found with display name: {display_name}")
            
            parent_model = models[0]
            
            # Use ModelRegistry for alias management
            model_registry = aiplatform.models.ModelRegistry(model=parent_model)

            # Get target version ID
            version_id = getattr(model, "version_id", None)

            print(f"Setting version {version_id} as default")

            # Add 'default' alias to the target version.
            # Vertex AI automatically transfers the alias from the old version -
            # you cannot explicitly remove the 'default' alias, only reassign it.
            model_registry.add_version_aliases(["default"], version_id)

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
    
    report_path = os.path.join(DOCS_DIR, "model_selection_report.json")    
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
    """
    import sys
    
    # Configuration
    improvement_threshold = 0.0  # Require any improvement
    
    manager = ModelManager(improvement_threshold=improvement_threshold)
    client = bigquery.Client()
    project_id = client.project
    dataset_id = "books"
    
    print("\n" + "="*80)
    print("AUTOMATIC MODEL ROLLBACK MANAGER")
    print("="*80)
    
    # 1. Determine Model and Tables
    print("\nChecking which model was selected...")
    selected_info = get_selected_model_from_report()
    
    if selected_info:
        model_name = selected_info['model_name']
        new_predictions_table = selected_info['predictions_table']
        model_display_name = get_vertex_ai_model_name(model_name)
        print(f"✓ Using selected model: {model_name}")
    else:
        model_name = "boosted_tree_regressor"
        model_display_name = "goodreads_boosted_tree_regressor"
        new_predictions_table = f"{project_id}.{dataset_id}.boosted_tree_rating_predictions"
        print(f"Using default model: {model_name}")

    print(f"Processing: {model_display_name}")
    
    # 2. Verify New Predictions Exist
    try:
        client.get_table(new_predictions_table)
        print(f"✓ Found new predictions table: {new_predictions_table}")
    except Exception:
        print(f"✗ ERROR: Predictions table not found: {new_predictions_table}")
        sys.exit(1)
    
    # 3. Get Model Versions
    versions = manager.get_model_versions(model_display_name)
    if not versions:
        print("✗ ERROR: No versions found.")
        sys.exit(1)
        
    new_model = versions[0]
    
    # Find current default
    current_default = None
    for version in versions:
        if hasattr(version, 'version_aliases') and 'default' in version.version_aliases:
            current_default = version
            break
    
    # Handle First Deployment
    if not current_default:
        print("\nNo current default found. Setting new model as default (First Deployment).")
        manager.set_default_version(new_model, model_display_name)
        sys.exit(0)
        
    if new_model.version_id == current_default.version_id:
        print("\nLatest version is already the default - no action needed.")
        sys.exit(0)
        
    print(f"\nNew Version: {new_model.version_id}")
    print(f"Current Default: {current_default.version_id}")

    # 4. Find Current Model Artifact (The Fix)
    print("\nLocating current default model artifact...")
    current_predictions_table = f"{project_id}.{dataset_id}.{model_name}_predictions_current"
    current_bqml_model = None
    artifact_uri = None
    
    try:
        # A) Try Vertex AI Metadata
        parents = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
        if parents:
            parent_model = parents[0]
            current_model_resource = aiplatform.Model(
                model_name=parent_model.resource_name,
                version=current_default.version_id
            )
            # Try standard property then fallback to dict
            artifact_uri = getattr(current_model_resource, 'uri', None) or \
                           getattr(current_model_resource, 'artifact_uri', None)
            if not artifact_uri:
                artifact_uri = current_model_resource.to_dict().get("artifactUri")
                
            if artifact_uri:
                print(f"  ✓ Found URI in metadata: {artifact_uri}")

        # B) Fallback: Timestamp Matching (Widened Window)
        if not artifact_uri:
            print("  ⚠ URI missing. Attempting to find BQ model by timestamp...")
            v_create_time = current_model_resource.create_time
            bq_prefix = f"{model_name}_model_" # e.g. boosted_tree_regressor_model_
            
            # Get all BQ models
            dataset_ref = client.dataset(dataset_id, project=project_id)
            bq_models = list(client.list_models(dataset_ref))
            
            best_match = None
            min_diff = float('inf')
            
            # Window: 24 hours (in seconds)
            SEARCH_WINDOW = 86400 
            
            for bq_model in bq_models:
                if bq_model.model_id.startswith(bq_prefix):
                    # Compare timestamps (absolute difference)
                    diff = abs(bq_model.created.timestamp() - v_create_time.timestamp())
                    if diff < SEARCH_WINDOW and diff < min_diff:
                        min_diff = diff
                        best_match = bq_model.model_id
            
            if best_match:
                print(f"  ✓ Found matching BQ model (diff {min_diff/60:.1f} min): {best_match}")
                artifact_uri = f"bq://{project_id}.{dataset_id}.{best_match}"
            else:
                print("  ✗ No matching timestamped model found.")
                
        # C) Fallback: Legacy Static Name (The New Logic)
        if not artifact_uri:
            print("  ⚠ Checking for legacy model (no timestamp)...")
            # Construct legacy name: e.g., boosted_tree_regressor_model
            # Based on bq_model_training.py, the base is "{model_name}_model"
            legacy_name = f"{model_name}_model"
            legacy_full_id = f"{project_id}.{dataset_id}.{legacy_name}"
            
            try:
                client.get_model(legacy_full_id)
                print(f"  ✓ Found legacy model: {legacy_full_id}")
                artifact_uri = f"bq://{legacy_full_id}"
            except Exception:
                print(f"  ✗ Legacy model not found: {legacy_full_id}")

        # D) Extract ID from URI
        if artifact_uri and artifact_uri.startswith('bq://'):
            current_bqml_model = artifact_uri.replace('bq://', '')

    except Exception as e:
        print(f"  Error locating current model: {e}")

    # 5. Generate Predictions OR Force Promote
    force_promote = False
    
    if current_bqml_model:
        # We have a model, generate predictions
        val_table = f"{project_id}.{dataset_id}.goodreads_test_set"
        if manager.generate_predictions_table(current_bqml_model, val_table, current_predictions_table):
            print("✓ Generated predictions for current default.")
        else:
            print("⚠ Failed to generate predictions for current model.")
            force_promote = True
    else:
        # We LOST the current model
        print("\n" + "!"*80)
        print("WARNING: Current default model is broken or missing!")
        print(f"  Could not locate BQ artifact for version {current_default.version_id}.")
        print("  Cannot perform comparison.")
        print("!"*80)
        force_promote = True

    # 6. Execution Decision
    if force_promote:
        print("\n> DECISION: Forcing promotion of NEW model.")
        print("  Reason: Current default is invalid/missing. We must restore a working default.")
        manager.set_default_version(new_model, model_display_name)
        print(f"\n✓ SUCCESS: Version {new_model.version_id} is now the default.")
        sys.exit(0)
    else:
        # Normal Comparison
        result = manager.manage_model_rollback(
            display_name=model_display_name,
            new_model_predictions_table=new_predictions_table,
            current_model_predictions_table=current_predictions_table
        )
        
        if result['decision'] in ['PROMOTED_NEW_VERSION', 'PROMOTED_FIRST_DEFAULT']:
            sys.exit(0)
        elif result['decision'] == 'ROLLBACK_KEPT_CURRENT':
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
