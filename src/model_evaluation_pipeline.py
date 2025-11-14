"""
Model Evaluation Pipeline for Goodreads Recommendation System

This module orchestrates model evaluation workflow:
1. Loads trained model predictions
2. Computes performance metrics (MAE, RMSE)
3. Performs feature importance analysis (SHAP)
4. Generates evaluation reports

This is SEPARATE from bias detection - focuses on model performance and interpretability.

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from datetime import datetime
from typing import Dict, Optional
import json
from google.cloud import bigquery
from model_sensitivity_analysis import ModelSensitivityAnalyzer


class ModelEvaluationPipeline:
    """
    Complete pipeline for model evaluation and interpretability.
    """

    def __init__(self, project_id: Optional[str] = None):
        """Initialize the evaluation pipeline."""
        airflow_home = os.environ.get("AIRFLOW_HOME", "")
        possible_paths = [
            os.path.join(airflow_home, "gcp_credentials.json") if airflow_home else None,
            "config/gcp_credentials.json",
            "gcp_credentials.json",
            "../config/gcp_credentials.json",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config",
                         "gcp_credentials.json")
        ]

        credentials_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                credentials_path = os.path.abspath(path)
                break

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.client = bigquery.Client(project=project_id)
        self.project_id = self.client.project
        self.dataset_id = "books"
        self.sensitivity_analyzer = ModelSensitivityAnalyzer(project_id=project_id)

        print(f"ModelEvaluationPipeline initialized for project: {self.project_id}")

    def evaluate_model(
            self,
            model_name: str,
            predictions_table: str,
            run_sensitivity_analysis: bool = True,
            sensitivity_sample_size: int = 1000
    ) -> Dict:
        """
        Run complete model evaluation.

        Args:
            model_name: Name of the model
            predictions_table: BigQuery table with predictions
            run_sensitivity_analysis: Whether to run SHAP analysis
            sensitivity_sample_size: Number of samples for SHAP

        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "=" * 80)
        print(f"MODEL EVALUATION PIPELINE: {model_name}")
        print("=" * 80 + "\n")

        evaluation_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions_table': predictions_table,
            'performance_metrics': None,
            'sensitivity_analysis': None
        }

        # Step 1: Compute Performance Metrics
        print("[STEP 1/2] Computing Performance Metrics...")
        performance = self._compute_performance_metrics(predictions_table)
        evaluation_results['performance_metrics'] = performance

        print(f"\n--- Performance Metrics ---")
        print(f"  Predictions: {performance['num_predictions']:,}")
        print(f"  MAE: {performance['mae']:.4f}")
        print(f"  RMSE: {performance['rmse']:.4f}")
        print(f"  Mean Predicted: {performance['mean_predicted']:.4f}")
        print(f"  Mean Actual: {performance['mean_actual']:.4f}")

        # Step 2: Feature Importance Analysis
        if run_sensitivity_analysis:
            print(f"\n[STEP 2/2] Running Feature Importance Analysis...")
            try:
                sensitivity_results = self.sensitivity_analyzer.analyze_feature_importance(
                    predictions_table=predictions_table,
                    model_name=model_name,
                    sample_size=sensitivity_sample_size
                )
                evaluation_results['sensitivity_analysis'] = sensitivity_results

                print(f"\n✓ Feature importance analysis complete")
                print(f"  Top 3 features:")
                for i, feat in enumerate(sensitivity_results['feature_importance'][:3], 1):
                    print(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")

            except Exception as e:
                print(f"Warning: Could not complete sensitivity analysis: {e}")
                evaluation_results['sensitivity_analysis'] = None
        else:
            print(f"\n[STEP 2/2] Skipping sensitivity analysis (disabled)")

        # Save Evaluation Report
        print(f"\nSaving evaluation report...")
        report_path = self._save_evaluation_report(evaluation_results, model_name)
        evaluation_results['report_path'] = report_path

        print("\n" + "=" * 80)
        print("MODEL EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nEvaluation report: {report_path}")
        if evaluation_results.get('sensitivity_analysis'):
            print(f"Feature importance: ../docs/model_analysis/sensitivity/")

        return evaluation_results

    def _compute_performance_metrics(self, predictions_table: str) -> Dict:
        """Compute overall performance metrics."""
        query = f"""
        SELECT 
            COUNT(*) as num_predictions,
            AVG(ABS(actual_rating - predicted_rating)) as mae,
            SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) as rmse,
            AVG(predicted_rating) as mean_predicted,
            AVG(actual_rating) as mean_actual,
            STDDEV(actual_rating - predicted_rating) as std_error
        FROM `{predictions_table}`
        WHERE actual_rating IS NOT NULL 
            AND predicted_rating IS NOT NULL
        """

        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return {
                'num_predictions': int(df['num_predictions'].iloc[0]),
                'mae': float(df['mae'].iloc[0]),
                'rmse': float(df['rmse'].iloc[0]),
                'mean_predicted': float(df['mean_predicted'].iloc[0]),
                'mean_actual': float(df['mean_actual'].iloc[0]),
                'std_error': float(df['std_error'].iloc[0]) if df['std_error'].iloc[0] else 0.0
            }
        except Exception as e:
            print(f"Error computing performance metrics: {e}")
            return {}

    def _save_evaluation_report(self, evaluation_results: Dict, model_name: str) -> str:
        """Save evaluation report to JSON."""
        output_dir = "../docs/model_analysis/evaluation"
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")

        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        print(f"  Evaluation report saved to: {report_path}")
        return report_path


def main():
    """Run model evaluation for all models."""
    pipeline = ModelEvaluationPipeline()

    models = [
        ("boosted_tree_regressor",
         f"{pipeline.project_id}.{pipeline.dataset_id}.boosted_tree_rating_predictions"),
        ("matrix_factorization",
         f"{pipeline.project_id}.{pipeline.dataset_id}.matrix_factorization_rating_predictions")
    ]

    for model_name, predictions_table in models:
        try:
            results = pipeline.evaluate_model(
                model_name=model_name,
                predictions_table=predictions_table,
                run_sensitivity_analysis=True,
                sensitivity_sample_size=1000
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    print("\n" + "=" * 80)
    print("ALL MODEL EVALUATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()