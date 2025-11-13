"""
Bias Mitigation Module for Goodreads Recommendation System

This module implements bias mitigation techniques including:
1. Re-weighting: Adjust training sample weights to balance group representation
2. Threshold Adjustment: Apply different decision thresholds per group
3. Calibration: Post-processing to calibrate predictions across groups
4. Shrinkage: Pull group means toward global mean to reduce bias

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict


@dataclass
class MitigationConfig:
    """Configuration for bias mitigation."""
    technique: str  # 'reweighting', 'threshold', 'calibration', 'shrinkage'
    target_dimensions: List[str]
    lambda_shrinkage: float = 0.5  # For shrinkage technique
    threshold_adjustments: Dict[str, float] = None  # For threshold technique
    reweight_strategy: str = 'inverse_propensity'  # 'inverse_propensity' or 'balanced'


@dataclass
class MitigationResult:
    """Results from bias mitigation."""
    technique: str
    timestamp: str
    original_metrics: Dict
    mitigated_metrics: Dict
    improvement_pct: Dict
    output_table: str


class BiasMitigator:
    """
    Implements various bias mitigation techniques for recommendation models.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the Bias Mitigator.
        
        Args:
            project_id: GCP project ID (if None, uses default from credentials)
        """
        airflow_home = os.environ.get("AIRFLOW_HOME", "")
        possible_paths = [
            os.path.join(airflow_home, "gcp_credentials.json") if airflow_home else None,
            "config/gcp_credentials.json",
            "gcp_credentials.json",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "gcp_credentials.json")
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
        
        print(f"BiasMitigator initialized for project: {self.project_id}")
    
    def apply_shrinkage_mitigation(
        self,
        features_table: str,
        output_table: str,
        slice_dimension: str,
        slice_expression: str,
        lambda_shrinkage: float = 0.5
    ) -> MitigationResult:
        """
        Apply shrinkage-based bias mitigation.
        
        This technique pulls group-specific rating means toward the global mean,
        reducing the influence of group bias on recommendations.
        
        Args:
            features_table: Input features table
            output_table: Output table with mitigated ratings
            slice_dimension: Name of the dimension (e.g., "Popularity")
            slice_expression: SQL expression to define the slice
            lambda_shrinkage: Shrinkage parameter (0=no shrinkage, 1=full shrinkage)
            
        Returns:
            MitigationResult with before/after metrics
        """
        print(f"\n{'='*80}")
        print(f"APPLYING SHRINKAGE MITIGATION: {slice_dimension}")
        print(f"Lambda: {lambda_shrinkage}")
        print(f"{'='*80}\n")
        
        # Compute original metrics
        original_metrics = self._compute_group_metrics(features_table, slice_expression)
        
        # Apply shrinkage
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        WITH base AS (
            SELECT *,
                   {slice_expression} AS slice_group
            FROM `{features_table}`
        ),
        group_stats AS (
            SELECT 
                slice_group,
                AVG(user_avg_rating_vs_book) AS group_mean,
                COUNT(*) AS group_count
            FROM base
            GROUP BY slice_group
        ),
        global_stats AS (
            SELECT AVG(user_avg_rating_vs_book) AS global_mean
            FROM base
        ),
        mitigated AS (
            SELECT
                b.*,
                g.group_mean,
                (SELECT global_mean FROM global_stats) AS global_mean,
                -- Apply shrinkage: adjusted_rating = original - λ * (group_mean - global_mean)
                b.user_avg_rating_vs_book - {lambda_shrinkage} * (g.group_mean - (SELECT global_mean FROM global_stats)) AS rating_debiased,
                -- Also adjust predicted ratings if they exist
                CASE 
                    WHEN b.rating IS NOT NULL THEN
                        b.rating - {lambda_shrinkage} * (g.group_mean - (SELECT global_mean FROM global_stats))
                    ELSE b.rating
                END AS rating_debiased_predicted
            FROM base b
            JOIN group_stats g ON b.slice_group = g.slice_group
        )
        SELECT 
            * EXCEPT(slice_group),
            slice_group AS {slice_dimension}_group
        FROM mitigated
        """
        
        try:
            print(f"Executing shrinkage mitigation query...")
            job = self.client.query(query)
            job.result()
            print(f"Mitigated data written to: {output_table}")
            
            # Compute mitigated metrics
            mitigated_metrics = self._compute_group_metrics(output_table, slice_expression)
            
            # Calculate improvements
            improvement_pct = self._calculate_improvements(original_metrics, mitigated_metrics)
            
            result = MitigationResult(
                technique="shrinkage",
                timestamp=datetime.now().isoformat(),
                original_metrics=original_metrics,
                mitigated_metrics=mitigated_metrics,
                improvement_pct=improvement_pct,
                output_table=output_table
            )
            
            self._print_mitigation_summary(result, slice_dimension)
            
            return result
            
        except Exception as e:
            print(f"Error applying shrinkage mitigation: {e}")
            raise
    
    def create_reweighted_training_table(
        self,
        training_table: str,
        output_table: str,
        slice_dimension: str,
        slice_expression: str,
        strategy: str = 'inverse_propensity'
    ) -> MitigationResult:
        """
        Create a re-weighted training table to balance group representation.
        
        Args:
            training_table: Input training table
            output_table: Output table with sample weights
            slice_dimension: Name of the dimension
            slice_expression: SQL expression to define the slice
            strategy: 'inverse_propensity' or 'balanced'
            
        Returns:
            MitigationResult with weight statistics
        """
        print(f"\n{'='*80}")
        print(f"CREATING RE-WEIGHTED TRAINING TABLE: {slice_dimension}")
        print(f"Strategy: {strategy}")
        print(f"{'='*80}\n")
        
        # Compute original metrics
        original_metrics = self._compute_group_metrics(training_table, slice_expression)
        
        if strategy == 'inverse_propensity':
            weight_formula = """
                1.0 / (CAST(group_count AS FLOAT64) / total_count)
            """
        else:  # balanced
            weight_formula = """
                (total_count / num_groups) / CAST(group_count AS FLOAT64)
            """
        
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        WITH base AS (
            SELECT *,
                   {slice_expression} AS slice_group
            FROM `{training_table}`
        ),
        group_counts AS (
            SELECT 
                slice_group,
                COUNT(*) AS group_count
            FROM base
            GROUP BY slice_group
        ),
        total AS (
            SELECT 
                COUNT(*) AS total_count,
                COUNT(DISTINCT slice_group) AS num_groups
            FROM base
        ),
        weighted AS (
            SELECT
                b.*,
                g.group_count,
                t.total_count,
                t.num_groups,
                {weight_formula} AS sample_weight
            FROM base b
            JOIN group_counts g ON b.slice_group = g.slice_group
            CROSS JOIN total t
        )
        SELECT 
            * EXCEPT(slice_group, group_count, total_count, num_groups),
            slice_group AS {slice_dimension}_group,
            sample_weight
        FROM weighted
        """
        
        try:
            print(f"Creating re-weighted training table...")
            job = self.client.query(query)
            job.result()
            print(f"Re-weighted training data written to: {output_table}")
            
            # Analyze weight distribution
            weight_stats = self._analyze_weights(output_table)
            
            result = MitigationResult(
                technique="reweighting",
                timestamp=datetime.now().isoformat(),
                original_metrics=original_metrics,
                mitigated_metrics=weight_stats,
                improvement_pct={},
                output_table=output_table
            )
            
            self._print_reweighting_summary(result, slice_dimension)
            
            return result
            
        except Exception as e:
            print(f"Error creating re-weighted training table: {e}")
            raise
    
    def apply_threshold_adjustments(
        self,
        predictions_table: str,
        output_table: str,
        slice_dimension: str,
        slice_expression: str,
        threshold_adjustments: Optional[Dict[str, float]] = None
    ) -> MitigationResult:
        """
        Apply group-specific threshold adjustments to predictions.
        
        Args:
            predictions_table: Input predictions table
            output_table: Output table with adjusted predictions
            slice_dimension: Name of the dimension
            slice_expression: SQL expression to define the slice
            threshold_adjustments: Dict mapping slice values to adjustment factors
            
        Returns:
            MitigationResult with adjustment statistics
        """
        print(f"\n{'='*80}")
        print(f"APPLYING THRESHOLD ADJUSTMENTS: {slice_dimension}")
        print(f"{'='*80}\n")
        
        # If no adjustments provided, compute optimal ones
        if threshold_adjustments is None:
            threshold_adjustments = self._compute_optimal_thresholds(
                predictions_table,
                slice_expression
            )
        
        # Build CASE statement for adjustments
        case_statements = []
        for slice_value, adjustment in threshold_adjustments.items():
            case_statements.append(
                f"WHEN slice_group = '{slice_value}' THEN predicted_rating + {adjustment}"
            )
        
        case_sql = "\n                    ".join(case_statements)
        
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        WITH base AS (
            SELECT *,
                   {slice_expression} AS slice_group
            FROM `{predictions_table}`
        )
        SELECT
            * EXCEPT(slice_group, predicted_rating),
            slice_group AS {slice_dimension}_group,
            predicted_rating AS original_predicted_rating,
            CASE
                {case_sql}
                ELSE predicted_rating
            END AS predicted_rating
        FROM base
        """
        
        try:
            print(f"Applying threshold adjustments...")
            print(f"Adjustments: {threshold_adjustments}")
            job = self.client.query(query)
            job.result()
            print(f"Adjusted predictions written to: {output_table}")
            
            # Compute metrics before and after
            original_metrics = self._compute_prediction_metrics(predictions_table)
            adjusted_metrics = self._compute_prediction_metrics(output_table)
            improvement_pct = self._calculate_improvements(original_metrics, adjusted_metrics)
            
            result = MitigationResult(
                technique="threshold_adjustment",
                timestamp=datetime.now().isoformat(),
                original_metrics=original_metrics,
                mitigated_metrics=adjusted_metrics,
                improvement_pct=improvement_pct,
                output_table=output_table
            )
            
            self._print_mitigation_summary(result, slice_dimension)
            
            return result
            
        except Exception as e:
            print(f"Error applying threshold adjustments: {e}")
            raise
    
    def _compute_group_metrics(self, table: str, slice_expression: str) -> Dict:
        """Compute group-level metrics."""
        query = f"""
        WITH base AS (
            SELECT *,
                   {slice_expression} AS slice_group
            FROM `{table}`
        )
        SELECT
            slice_group,
            COUNT(*) AS count,
            AVG(user_avg_rating_vs_book) AS mean_rating,
            STDDEV(user_avg_rating_vs_book) AS std_rating
        FROM base
        WHERE slice_group IS NOT NULL
        GROUP BY slice_group
        ORDER BY slice_group
        """
        
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            metrics = {}
            for _, row in df.iterrows():
                metrics[row['slice_group']] = {
                    'count': int(row['count']),
                    'mean_rating': float(row['mean_rating']),
                    'std_rating': float(row['std_rating']) if pd.notna(row['std_rating']) else 0.0
                }
            return metrics
        except Exception as e:
            print(f"Error computing group metrics: {e}")
            return {}
    
    def _compute_prediction_metrics(self, table: str) -> Dict:
        """Compute prediction metrics."""
        query = f"""
        SELECT
            COUNT(*) AS count,
            AVG(ABS(actual_rating - predicted_rating)) AS mae,
            SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) AS rmse
        FROM `{table}`
        WHERE actual_rating IS NOT NULL AND predicted_rating IS NOT NULL
        """
        
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return {
                'count': int(df['count'].iloc[0]),
                'mae': float(df['mae'].iloc[0]),
                'rmse': float(df['rmse'].iloc[0])
            }
        except Exception as e:
            print(f"Error computing prediction metrics: {e}")
            return {}
    
    def _compute_optimal_thresholds(
        self,
        predictions_table: str,
        slice_expression: str
    ) -> Dict[str, float]:
        """Compute optimal threshold adjustments to minimize MAE per group."""
        query = f"""
        WITH base AS (
            SELECT *,
                   {slice_expression} AS slice_group
            FROM `{predictions_table}`
            WHERE actual_rating IS NOT NULL AND predicted_rating IS NOT NULL
        )
        SELECT
            slice_group,
            AVG(actual_rating - predicted_rating) AS mean_error
        FROM base
        WHERE slice_group IS NOT NULL
        GROUP BY slice_group
        """
        
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            adjustments = {}
            for _, row in df.iterrows():
                adjustments[row['slice_group']] = float(row['mean_error'])
            return adjustments
        except Exception as e:
            print(f"Error computing optimal thresholds: {e}")
            return {}
    
    def _analyze_weights(self, table: str) -> Dict:
        """Analyze weight distribution in re-weighted table."""
        query = f"""
        SELECT
            MIN(sample_weight) AS min_weight,
            MAX(sample_weight) AS max_weight,
            AVG(sample_weight) AS mean_weight,
            STDDEV(sample_weight) AS std_weight
        FROM `{table}`
        """
        
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return {
                'min_weight': float(df['min_weight'].iloc[0]),
                'max_weight': float(df['max_weight'].iloc[0]),
                'mean_weight': float(df['mean_weight'].iloc[0]),
                'std_weight': float(df['std_weight'].iloc[0])
            }
        except Exception as e:
            print(f"Error analyzing weights: {e}")
            return {}
    
    def _calculate_improvements(self, original: Dict, mitigated: Dict) -> Dict:
        """Calculate percentage improvements."""
        improvements = {}
        
        # For group metrics
        if all(isinstance(v, dict) for v in original.values()):
            original_variance = np.var([v['mean_rating'] for v in original.values()])
            mitigated_variance = np.var([v['mean_rating'] for v in mitigated.values()])
            improvements['variance_reduction_pct'] = (
                (original_variance - mitigated_variance) / original_variance * 100
                if original_variance > 0 else 0
            )
        
        # For prediction metrics
        if 'mae' in original and 'mae' in mitigated:
            improvements['mae_improvement_pct'] = (
                (original['mae'] - mitigated['mae']) / original['mae'] * 100
                if original['mae'] > 0 else 0
            )
            improvements['rmse_improvement_pct'] = (
                (original['rmse'] - mitigated['rmse']) / original['rmse'] * 100
                if original['rmse'] > 0 else 0
            )
        
        return improvements
    
    def _print_mitigation_summary(self, result: MitigationResult, dimension: str):
        """Print summary of mitigation results."""
        print(f"\n{'='*80}")
        print(f"MITIGATION SUMMARY: {result.technique.upper()}")
        print(f"{'='*80}\n")
        print(f"Dimension: {dimension}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Output: {result.output_table}")
        
        if result.improvement_pct:
            print("\n--- Improvements ---")
            for metric, value in result.improvement_pct.items():
                print(f"  {metric}: {value:+.2f}%")
        
        print(f"\n{'='*80}\n")
    
    def _print_reweighting_summary(self, result: MitigationResult, dimension: str):
        """Print summary of re-weighting results."""
        print(f"\n{'='*80}")
        print(f"RE-WEIGHTING SUMMARY")
        print(f"{'='*80}\n")
        print(f"Dimension: {dimension}")
        print(f"Output: {result.output_table}")
        
        print("\n--- Weight Statistics ---")
        for key, value in result.mitigated_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n{'='*80}\n")
    
    def save_mitigation_report(self, result: MitigationResult, output_path: str):
        """Save mitigation report to JSON file."""
        report_dict = asdict(result)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Mitigation report saved to: {output_path}")


def main():
    """Example usage of BiasMitigator."""
    mitigator = BiasMitigator()
    
    # Example 1: Apply shrinkage mitigation
    features_table = f"{mitigator.project_id}.{mitigator.dataset_id}.goodreads_features"
    output_table = f"{mitigator.project_id}.{mitigator.dataset_id}.goodreads_features_debiased"
    
    result = mitigator.apply_shrinkage_mitigation(
        features_table=features_table,
        output_table=output_table,
        slice_dimension="Popularity",
        slice_expression="""
            CASE
                WHEN book_popularity_normalized >= 0.66 THEN 'High'
                WHEN book_popularity_normalized >= 0.33 THEN 'Medium'
                ELSE 'Low'
            END
        """,
        lambda_shrinkage=0.5
    )
    
    mitigator.save_mitigation_report(
        result,
        "data/bias_reports/shrinkage_mitigation_report.json"
    )


if __name__ == "__main__":
    main()
