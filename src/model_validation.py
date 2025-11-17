"""
BigQuery ML Validation Module
Evaluates a trained model on train/val/test splits using ML.EVALUATE.
Logs metrics to MLflow, saves JSON reports, and returns pass/fail status.
"""

import os
import json
from datetime import datetime
from google.cloud import bigquery
import mlflow
import pandas as pd


class BigQueryModelValidator:
    def __init__(self, project_id=None, dataset_id="books"):
        """Initialize BigQuery client with credential auto-discovery."""

        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = airflow_home + "/gcp_credentials.json"
      
        self.client = bigquery.Client(project=project_id)
        self.project_id = self.client.project
        self.dataset_id = dataset_id

        # BigQuery dataset tables
        self.train_table = f"{self.project_id}.{dataset_id}.goodreads_train_set"
        self.val_table = f"{self.project_id}.{dataset_id}.goodreads_validation_set"
        self.test_table = f"{self.project_id}.{dataset_id}.goodreads_test_set"

        print(f"BigQueryModelValidator initialized for project: {self.project_id}")

    # ------------------------------------------------------------------
    # Evaluate train / val / test split
    # ------------------------------------------------------------------
    def evaluate_split(self, model_name: str, split: str, table: str) -> pd.DataFrame:
        print(f"\nEvaluating {model_name} on {split}...")

        query = f"""
        SELECT *
        FROM ML.EVALUATE(
            MODEL `{model_name}`,
            (SELECT * FROM `{table}` WHERE rating IS NOT NULL)
        )
        """

        df = self.client.query(query).to_dataframe(create_bqstorage_client=False)

        if df.empty:
            print(f"No evaluation results for split '{split}'.")
        else:
            print(f"\n{split.upper()} Evaluation Metrics:")
            print(df.T)

        return df

    # ------------------------------------------------------------------
    # Log metrics to MLflow
    # ------------------------------------------------------------------
    def log_mlflow_metrics(self, df, split, model_prefix):
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, (float, int)):
                mlflow.log_metric(f"{model_prefix}_{split}_{col}", float(val))

    # ------------------------------------------------------------------
    # JSON saving
    # ------------------------------------------------------------------
    def save_json_report(self, model_label: str, model_name: str, results: dict):
        """Save validation results as JSON inside ../docs/model_validation_report/"""

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, "docs", "model_validation_report")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"{model_label}_validation_{timestamp}.json")

        print("Saving JSON to:", file_path)

        serializable_results = {}
        for split, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                serializable_results[split] = df.to_dict(orient="records")[0]
            else:
                serializable_results[split] = {}

        output = {
            "model_label": model_label,
            "model_name": model_name,
            "timestamp": timestamp,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "metrics": serializable_results
        }

        with open(file_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"JSON saved successfully: {file_path}")

    # ------------------------------------------------------------------
    # Threshold-based validation
    # ------------------------------------------------------------------
    def validate_model(self, model_name: str, model_label: str):
        print("\n" + "=" * 80)
        print(f"     VALIDATION REPORT FOR MODEL: {model_label}")
        print("=" * 80)

        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("goodreads_model_validation")

        with mlflow.start_run(run_name=f"validate_{model_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_label", model_label)

    
            splits = {
                "train": self.train_table,
                "val": self.val_table,
                "test": self.test_table
            }

            results = {}
            for split, table in splits.items():
                df = self.evaluate_split(model_name, split, table)
                results[split] = df

                if not df.empty:
                    self.log_mlflow_metrics(df, split, model_label)


            self.save_json_report(model_label, model_name, results)

            # -----------------------------
            # VALIDATION THRESHOLD CHECK
            # -----------------------------
            val_df = results["val"]

            print("\nDEBUG: Validation DF columns:", val_df.columns.tolist())

            # Auto-detect correct RMSE column from BigQuery ML
            if "rmse" in val_df.columns:
                rmse_col = "rmse"
            elif "root_mean_squared_error" in val_df.columns:
                rmse_col = "root_mean_squared_error"
            elif "mean_squared_error" in val_df.columns:
                rmse_col = "mean_squared_error"
            else:
                raise KeyError(
                    f"No RMSE-like column found. Available columns: {val_df.columns.tolist()}"
                )

            rmse = float(val_df[rmse_col].iloc[0])
            rmse_threshold = 3.0 

            print(f"\nValidation RMSE ({rmse_col}) = {rmse:.4f}")
            print(f"Threshold RMSE = {rmse_threshold:.4f}")

            if rmse <= rmse_threshold:
                print("MODEL APPROVED")
                return True
            else:
                print("MODEL REJECTED — RMSE too high")
                return False
    
    def get_selected_model_from_report():
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
    
    def validate_selected_model(self):
        """
        Validate the model selected in the model selection report.
        If no report is found, validate the default boosted_tree_regressor model.
        """
        selected_model = self.get_selected_model_from_report()
        
        if selected_model:
            model_name = selected_model['model_name']
            model_label = "selected_model"
        else:
            # Default to boosted_tree_regressor model
            model_name = f"{self.project_id}.books.boosted_tree_regressor_model"
            model_label = "boosted_tree"
        
        return self.validate_model(model_name, model_label)



if __name__ == "__main__":
    validator = BigQueryModelValidator(project_id=None)

    ok = validator.validate_selected_model()

    if not (ok):
        print("Validation FAILED — stopping pipeline.")
        exit(1)

    print("All models VALIDATED successfully — pipeline may continue.")
    exit(0)
