"""
Bias Detection Module for Goodreads Recommendation System

This module performs comprehensive bias detection by:
1. Slicing predictions by demographic/feature groups
2. Computing performance metrics (MAE, RMSE, Accuracy) per slice
3. Identifying significant disparities across groups
4. Generating detailed bias reports

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class SliceMetrics:
    """Metrics for a single slice."""
    slice_name: str
    slice_dimension: str
    slice_value: str
    count: int
    mae: float
    rmse: float
    mean_predicted: float
    mean_actual: float
    mean_error: float
    std_error: float


@dataclass
class BiasReport:
    """Comprehensive bias report across all slices."""
    timestamp: str
    model_name: str
    dataset: str
    slice_metrics: List[SliceMetrics]
    disparity_analysis: Dict
    recommendations: List[str]


class BiasDetector:
    """
    Detects bias in recommendation model predictions across different demographic slices.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the Bias Detector.
        
        Args:
            project_id: GCP project ID (if None, uses default from credentials)
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"        
        self.client = bigquery.Client(project=project_id)
        self.project_id = self.client.project
        self.dataset_id = "books"
        
        print(f"BiasDetector initialized for project: {self.project_id}")
    
    def get_slice_definitions(self) -> List[Tuple[str, str, str]]:
        """
        Define the slices to analyze for bias.
        
        Returns:
            List of tuples (dimension_name, sql_expression, slice_label)
        """
        return [
            # Book Popularity
            ("Popularity", """
                CASE
                    WHEN book_popularity_normalized >= 0.66 THEN 'High'
                    WHEN book_popularity_normalized >= 0.33 THEN 'Medium'
                    ELSE 'Low'
                END
            """, "popularity_group"),
            
            # Book Length
            ("Book Length", "book_length_category", "length_group"),
            
            # Book Era
            ("Book Era", "book_era", "era_group"),
            
            # Number of Genres
            ("Genre Diversity", """
                CASE
                    WHEN num_genres >= 5 THEN 'Multi-genre'
                    WHEN num_genres >= 2 THEN 'Some genres'
                    ELSE 'Single/No genre'
                END
            """, "genre_group"),
            
            # User Activity Level
            ("User Activity", """
                CASE
                    WHEN user_activity_count >= 50 THEN 'High'
                    WHEN user_activity_count >= 10 THEN 'Medium'
                    ELSE 'Low'
                END
            """, "activity_group"),
            
            # Reading Pace
            ("Reading Pace", "reading_pace_category", "pace_group"),
            
            # Book Rating Range (actual rating)
            ("Rating Range", """
                CASE
                    WHEN actual_rating  >= 4.0 THEN 'High (4-5)'
                    WHEN actual_rating  >= 3.0 THEN 'Medium (3-4)'
                    ELSE 'Low (1-3)'
                END
            """, "rating_group"),
        ]
    
    def compute_slice_metrics(
        self,
        predictions_table: str,
        slice_dimension: str,
        slice_expression: str,
        slice_label: str
    ) -> List[SliceMetrics]:
        """
        Compute performance metrics for each slice within a dimension.
        
        Args:
            predictions_table: Full path to predictions table
            slice_dimension: Human-readable name of the dimension
            slice_expression: SQL expression to define the slice
            slice_label: Column name for the slice group
            
        Returns:
            List of SliceMetrics for each group in the dimension
        """
        query = f"""
        WITH sliced_predictions AS (
            SELECT
                {slice_expression} AS slice_group,
                actual_rating,
                predicted_rating,
                ABS(actual_rating - predicted_rating) AS absolute_error,
                (actual_rating - predicted_rating) AS error
            FROM `{predictions_table}`
            WHERE actual_rating IS NOT NULL
                AND predicted_rating IS NOT NULL
        )
        SELECT
            slice_group,
            COUNT(*) AS count,
            AVG(absolute_error) AS mae,
            SQRT(AVG(POWER(error, 2))) AS rmse,
            AVG(predicted_rating) AS mean_predicted,
            AVG(actual_rating) AS mean_actual,
            AVG(error) AS mean_error,
            STDDEV(error) AS std_error
        FROM sliced_predictions
        WHERE slice_group IS NOT NULL
        GROUP BY slice_group
        ORDER BY slice_group
        """
        
        try:
            results = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            
            metrics = []
            for _, row in results.iterrows():
                metrics.append(SliceMetrics(
                    slice_name=f"{slice_dimension}={row['slice_group']}",
                    slice_dimension=slice_dimension,
                    slice_value=str(row['slice_group']),
                    count=int(row['count']),
                    mae=float(row['mae']),
                    rmse=float(row['rmse']),
                    mean_predicted=float(row['mean_predicted']),
                    mean_actual=float(row['mean_actual']),
                    mean_error=float(row['mean_error']),
                    std_error=float(row['std_error']) if pd.notna(row['std_error']) else 0.0
                ))
            
            return metrics
            
        except Exception as e:
            print(f"Error computing metrics for {slice_dimension}: {e}")
            return []
    
    def analyze_disparities(self, all_metrics: List[SliceMetrics]) -> Dict:
        """
        Analyze disparities across slices to identify potential bias.
        
        Args:
            all_metrics: List of all slice metrics
            
        Returns:
            Dictionary containing disparity analysis
        """
        # Group metrics by dimension
        dimensions = {}
        for metric in all_metrics:
            if metric.slice_dimension not in dimensions:
                dimensions[metric.slice_dimension] = []
            dimensions[metric.slice_dimension].append(metric)
        
        disparity_analysis = {
            "summary": {},
            "detailed_disparities": [],
            "high_risk_slices": []
        }
        
        # Analyze each dimension
        for dim_name, dim_metrics in dimensions.items():
            if len(dim_metrics) < 2:
                continue
            
            maes = [m.mae for m in dim_metrics]
            rmses = [m.rmse for m in dim_metrics]
            
            mae_range = max(maes) - min(maes)
            rmse_range = max(rmses) - min(rmses)
            mae_cv = np.std(maes) / np.mean(maes) if np.mean(maes) > 0 else 0
            
            disparity_analysis["summary"][dim_name] = {
                "mae_range": float(mae_range),
                "rmse_range": float(rmse_range),
                "mae_coefficient_of_variation": float(mae_cv),
                "num_slices": len(dim_metrics),
                "max_mae_slice": max(dim_metrics, key=lambda x: x.mae).slice_value,
                "min_mae_slice": min(dim_metrics, key=lambda x: x.mae).slice_value
            }
            
            # Flag high disparity dimensions (CV > 0.15 or range > 0.3)
            if mae_cv > 0.15 or mae_range > 0.3:
                disparity_analysis["detailed_disparities"].append({
                    "dimension": dim_name,
                    "severity": "high" if mae_cv > 0.25 else "medium",
                    "mae_range": float(mae_range),
                    "mae_cv": float(mae_cv),
                    "recommendation": f"Significant performance disparity detected in {dim_name}. Consider bias mitigation."
                })
            
            # Identify high-risk slices (MAE > 20% above dimension average)
            avg_mae = np.mean(maes)
            for metric in dim_metrics:
                if metric.mae > avg_mae * 1.2:
                    disparity_analysis["high_risk_slices"].append({
                        "slice": metric.slice_name,
                        "dimension": dim_name,
                        "mae": float(metric.mae),
                        "mae_deviation_pct": float((metric.mae - avg_mae) / avg_mae * 100),
                        "count": metric.count
                    })
        
        return disparity_analysis
    
    def generate_recommendations(self, disparity_analysis: Dict) -> List[str]:
        """
        Generate actionable recommendations based on bias analysis.
        
        Args:
            disparity_analysis: Results from analyze_disparities
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for high-risk dimensions
        if disparity_analysis["detailed_disparities"]:
            high_severity = [d for d in disparity_analysis["detailed_disparities"] if d["severity"] == "high"]
            if high_severity:
                recommendations.append(
                    f"HIGH PRIORITY: {len(high_severity)} dimensions show severe performance disparities. "
                    f"Dimensions: {', '.join([d['dimension'] for d in high_severity])}"
                )
            
            medium_severity = [d for d in disparity_analysis["detailed_disparities"] if d["severity"] == "medium"]
            if medium_severity:
                recommendations.append(
                    f"MEDIUM PRIORITY: {len(medium_severity)} dimensions show moderate disparities. "
                    f"Dimensions: {', '.join([d['dimension'] for d in medium_severity])}"
                )
        
        # Check for high-risk slices
        if disparity_analysis["high_risk_slices"]:
            top_risk = sorted(disparity_analysis["high_risk_slices"], 
                             key=lambda x: x["mae_deviation_pct"], 
                             reverse=True)[:3]
            recommendations.append(
                f"Target these high-error slices for mitigation: " +
                ", ".join([f"{s['slice']} (MAE: {s['mae']:.3f})" for s in top_risk])
            )
        
        # Specific mitigation recommendations
        for dim_name, summary in disparity_analysis["summary"].items():
            if summary["mae_coefficient_of_variation"] > 0.2:
                recommendations.append(
                    f"{dim_name}: Consider re-weighting training data or adjusting decision thresholds "
                    f"to balance performance across {summary['num_slices']} groups"
                )
        
        if not recommendations:
            recommendations.append("No significant bias detected across analyzed dimensions.")
        
        return recommendations
    
    def detect_bias(
        self,
        predictions_table: str,
        model_name: str,
        dataset: str = "test"
    ) -> BiasReport:
        """
        Run comprehensive bias detection on model predictions.
        
        Args:
            predictions_table: Full path to BigQuery predictions table
            model_name: Name of the model being evaluated
            dataset: Dataset being evaluated (train/val/test)
            
        Returns:
            BiasReport object with complete analysis
        """
        print("=" * 80)
        print(f"BIAS DETECTION: {model_name} on {dataset} dataset")
        print("=" * 80)
        
        all_metrics = []
        slice_definitions = self.get_slice_definitions()
        
        print(f"\nAnalyzing {len(slice_definitions)} dimensions...")
        
        for dim_name, slice_expr, slice_label in slice_definitions:
            print(f"  Processing: {dim_name}...", end=" ")
            metrics = self.compute_slice_metrics(
                predictions_table,
                dim_name,
                slice_expr,
                slice_label
            )
            all_metrics.extend(metrics)
            print(f"({len(metrics)} slices)")
        
        print(f"\nComputed metrics for {len(all_metrics)} total slices")
        
        print("\nAnalyzing disparities...")
        disparity_analysis = self.analyze_disparities(all_metrics)
        
        print("\nGenerating recommendations...")
        recommendations = self.generate_recommendations(disparity_analysis)
        
        report = BiasReport(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            dataset=dataset,
            slice_metrics=all_metrics,
            disparity_analysis=disparity_analysis,
            recommendations=recommendations
        )
        
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report: BiasReport):
        """Print a summary of the bias report."""
        print("\n" + "=" * 80)
        print("BIAS DETECTION SUMMARY")
        print("=" * 80)
        
        print(f"\nModel: {report.model_name}")
        print(f"Dataset: {report.dataset}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Slices Analyzed: {len(report.slice_metrics)}")
        
        print("\n--- Performance Disparities ---")
        if report.disparity_analysis["detailed_disparities"]:
            for disp in report.disparity_analysis["detailed_disparities"]:
                print(f"  • {disp['dimension']} [{disp['severity'].upper()}]:")
                print(f"    MAE Range: {disp['mae_range']:.4f}, CV: {disp['mae_cv']:.4f}")
        else:
            print("  No significant disparities detected")
        
        print("\n--- High-Risk Slices ---")
        if report.disparity_analysis["high_risk_slices"]:
            for risk in report.disparity_analysis["high_risk_slices"][:5]:
                print(f"  • {risk['slice']}: MAE={risk['mae']:.4f} "
                      f"(+{risk['mae_deviation_pct']:.1f}% above average)")
        else:
            print("  No high-risk slices identified")
        
        print("\n--- Recommendations ---")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, report: BiasReport, output_path: str):
        """
        Save bias report to JSON file.
        
        Args:
            report: BiasReport object
            output_path: Path to save JSON file
        """
        report_dict = {
            "timestamp": report.timestamp,
            "model_name": report.model_name,
            "dataset": report.dataset,
            "slice_metrics": [asdict(m) for m in report.slice_metrics],
            "disparity_analysis": report.disparity_analysis,
            "recommendations": report.recommendations
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\nBias report saved to: {output_path}")
    
    def create_bias_metrics_table(
        self,
        report: BiasReport,
        output_table: str
    ):
        """
        Save slice metrics to a BigQuery table for downstream analysis.
        
        Args:
            report: BiasReport object
            output_table: Full path to output BigQuery table
        """
        try:
            # Convert to DataFrame
            metrics_data = [asdict(m) for m in report.slice_metrics]
            df = pd.DataFrame(metrics_data)
             # If no metrics (e.g., missing predictions table), skip writing
            if df.empty:
                print(f"No slice metrics to save for {report.model_name}; skipping BigQuery write")
                return
            # Add metadata
            df['model_name'] = report.model_name
            df['dataset'] = report.dataset
            df['analysis_timestamp'] = report.timestamp
            
            # Write to BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                schema=[
                    bigquery.SchemaField("slice_name", "STRING"),
                    bigquery.SchemaField("slice_dimension", "STRING"),
                    bigquery.SchemaField("slice_value", "STRING"),
                    bigquery.SchemaField("count", "INTEGER"),
                    bigquery.SchemaField("mae", "FLOAT"),
                    bigquery.SchemaField("rmse", "FLOAT"),
                    bigquery.SchemaField("mean_predicted", "FLOAT"),
                    bigquery.SchemaField("mean_actual", "FLOAT"),
                    bigquery.SchemaField("mean_error", "FLOAT"),
                    bigquery.SchemaField("std_error", "FLOAT"),
                    bigquery.SchemaField("model_name", "STRING"),
                    bigquery.SchemaField("dataset", "STRING"),
                    bigquery.SchemaField("analysis_timestamp", "TIMESTAMP"),
                ]
            )
            
            job = self.client.load_table_from_dataframe(df, output_table, job_config=job_config)
            job.result()
            
            print(f"Bias metrics table created: {output_table}")
            
        except Exception as e:
            print(f"Error creating bias metrics table: {e}")


def main():
    """Example usage of BiasDetector."""
    detector = BiasDetector()
    
    # Example: Detect bias in boosted tree predictions
    predictions_table = f"{detector.project_id}.{detector.dataset_id}.boosted_tree_rating_predictions"
    
    report = detector.detect_bias(
        predictions_table=predictions_table,
        model_name="boosted_tree_regressor",
        dataset="test"
    )
    
    # Save report
    detector.save_report(report, "../docs/bias_reports/boosted_tree_bias_report.json")
    
    # Save to BigQuery
    output_table = f"{detector.project_id}.{detector.dataset_id}.bias_metrics_boosted_tree"
    detector.create_bias_metrics_table(report, output_table)


if __name__ == "__main__":
    main()
