"""
BigQuery ML Model Prediction Module for Goodreads Recommendation System

This module uses trained BigQuery ML models to make predictions:
1. MATRIX_FACTORIZATION - Generate recommendations for users
2. AUTOML_REGRESSOR - Predict ratings for user-book pairs
3. BOOSTED_TREE_REGRESSOR - Predict ratings with feature-rich approach

The predictions are stored in BigQuery tables for downstream consumption.

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger
import time


class BigQueryMLPredictions:
    """
    Class for making predictions using trained BigQuery ML models.
    """

    def __init__(self):
        """
        Initialize the BigQuery ML Predictions class.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration
        - BigQuery client and project information
        - Model and prediction table references
        """
        # Set Google Application Credentials for BigQuery access
        # Check multiple possible locations for credentials file
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
        else:
            # Fallback to default location
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(airflow_home, "gcp_credentials.json") if airflow_home else "config/gcp_credentials.json"

        # Initialize logging
        self.logger = get_logger("model_predictions")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # Define table references
        self.test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"
        self.features_table = f"{self.project_id}.{self.dataset_id}.goodreads_features"

        # Model table references
        self.matrix_factorization_model = f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
        self.automl_regressor_model = f"{self.project_id}.{self.dataset_id}.automl_regressor_model"
        self.boosted_tree_model = f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"

        # Prediction table references
        self.mf_recommendations_table = f"{self.project_id}.{self.dataset_id}.mf_recommendations"
        self.automl_predictions_table = f"{self.project_id}.{self.dataset_id}.automl_rating_predictions"
        self.boosted_tree_predictions_table = f"{self.project_id}.{self.dataset_id}.boosted_tree_rating_predictions"
        self.combined_predictions_table = f"{self.project_id}.{self.dataset_id}.combined_predictions"

    def get_feature_columns(self):
        """
        Get the list of feature columns to use for predictions.
        
        Returns:
            list: List of feature column names
        """
        try:
            table = self.client.get_table(self.features_table)
            all_columns = [field.name for field in table.schema]
            
            # Exclude non-feature columns
            exclude_columns = {
                'user_id_clean', 'book_id', 'rating', 'is_read',
                'user_days_to_read', 'user_book_recency'
            }
            
            feature_columns = [col for col in all_columns if col not in exclude_columns]
            self.logger.info(f"Found {len(feature_columns)} feature columns")
            return feature_columns
        except Exception as e:
            self.logger.error(f"Error getting feature columns: {e}", exc_info=True)
            return [
                'num_books_read', 'avg_rating_given', 'user_activity_count',
                'recent_activity_days', 'user_avg_reading_time_days',
                'average_rating', 'ratings_count', 'text_reviews_count',
                'log_ratings_count', 'popularity_score', 'title_length_in_characters',
                'title_length_in_words', 'description_length', 'num_genres',
                'is_series', 'num_pages', 'publication_year', 'book_age_years',
                'avg_book_reading_time_days', 'num_readers_with_reading_time',
                'adjusted_average_rating', 'great', 'book_popularity_normalized',
                'reading_pace_category'
            ]

    def generate_matrix_factorization_recommendations(self, top_k=10):
        """
        Generate recommendations using MATRIX_FACTORIZATION model.
        
        Args:
            top_k (int): Number of top recommendations per user
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Generating MATRIX_FACTORIZATION Recommendations")
            self.logger.info("=" * 60)

            # Get unique users from test set
            # Fixed: Use subquery to calculate rank first, then filter
            query = f"""
            CREATE OR REPLACE TABLE `{self.mf_recommendations_table}` AS
            SELECT
                user_id_clean,
                book_id,
                predicted_rating,
                rank
            FROM (
                SELECT
                    user_id_clean,
                    book_id,
                    predicted_rating,
                    ROW_NUMBER() OVER (PARTITION BY user_id_clean ORDER BY predicted_rating DESC) as rank
                FROM ML.RECOMMEND(MODEL `{self.matrix_factorization_model}`, (
                    SELECT DISTINCT user_id_clean
                    FROM `{self.test_table}`
                ))
            )
            WHERE rank <= {top_k}
            ORDER BY user_id_clean, rank
            """

            self.logger.info(f"Generating recommendations for users...")
            self.logger.info(f"Model: {self.matrix_factorization_model}")
            self.logger.info(f"Output: {self.mf_recommendations_table}")
            self.logger.info(f"Top K: {top_k}")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(DISTINCT user_id_clean) as num_users,
                COUNT(*) as total_recommendations,
                AVG(predicted_rating) as avg_predicted_rating
            FROM `{self.mf_recommendations_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            self.logger.info("MATRIX_FACTORIZATION Recommendations Generated:")
            self.logger.info(f"  Users: {stats['num_users'].iloc[0]}")
            self.logger.info(f"  Total Recommendations: {stats['total_recommendations'].iloc[0]}")
            self.logger.info(f"  Avg Predicted Rating: {stats['avg_predicted_rating'].iloc[0]:.4f}")

        except Exception as e:
            self.logger.error(f"Error generating MATRIX_FACTORIZATION recommendations: {e}", exc_info=True)
            raise

    def generate_automl_predictions(self):
        """
        Generate rating predictions using AUTOML_REGRESSOR model.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Generating AUTOML_REGRESSOR Predictions")
            self.logger.info("=" * 60)

            feature_columns = self.get_feature_columns()
            feature_list = ", ".join(feature_columns)

            query = f"""
            CREATE OR REPLACE TABLE `{self.automl_predictions_table}` AS
            SELECT
                user_id_clean,
                book_id,
                rating as actual_rating,
                predicted_rating,
                ABS(rating - predicted_rating) as absolute_error,
                (rating - predicted_rating) as error
            FROM ML.PREDICT(MODEL `{self.automl_regressor_model}`, (
                SELECT
                    {feature_list},
                    rating,
                    user_id_clean,
                    book_id
                FROM `{self.test_table}`
                WHERE rating IS NOT NULL
            ))
            """

            self.logger.info(f"Generating predictions on test set...")
            self.logger.info(f"Model: {self.automl_regressor_model}")
            self.logger.info(f"Output: {self.automl_predictions_table}")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as num_predictions,
                AVG(absolute_error) as mean_absolute_error,
                SQRT(AVG(POWER(error, 2))) as root_mean_squared_error,
                AVG(predicted_rating) as avg_predicted_rating,
                AVG(actual_rating) as avg_actual_rating
            FROM `{self.automl_predictions_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            self.logger.info("AUTOML_REGRESSOR Predictions Generated:")
            self.logger.info(f"  Predictions: {stats['num_predictions'].iloc[0]}")
            self.logger.info(f"  MAE: {stats['mean_absolute_error'].iloc[0]:.4f}")
            self.logger.info(f"  RMSE: {stats['root_mean_squared_error'].iloc[0]:.4f}")
            self.logger.info(f"  Avg Predicted: {stats['avg_predicted_rating'].iloc[0]:.4f}")
            self.logger.info(f"  Avg Actual: {stats['avg_actual_rating'].iloc[0]:.4f}")

        except Exception as e:
            self.logger.error(f"Error generating AUTOML_REGRESSOR predictions: {e}", exc_info=True)
            raise

    def generate_boosted_tree_predictions(self):
        """
        Generate rating predictions using BOOSTED_TREE_REGRESSOR model.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Generating BOOSTED_TREE_REGRESSOR Predictions")
            self.logger.info("=" * 60)

            feature_columns = self.get_feature_columns()
            feature_list = ", ".join(feature_columns)

            query = f"""
            CREATE OR REPLACE TABLE `{self.boosted_tree_predictions_table}` AS
            SELECT
                user_id_clean,
                book_id,
                rating as actual_rating,
                predicted_rating,
                ABS(rating - predicted_rating) as absolute_error,
                (rating - predicted_rating) as error
            FROM ML.PREDICT(MODEL `{self.boosted_tree_model}`, (
                SELECT
                    {feature_list},
                    rating,
                    user_id_clean,
                    book_id
                FROM `{self.test_table}`
                WHERE rating IS NOT NULL
            ))
            """

            self.logger.info(f"Generating predictions on test set...")
            self.logger.info(f"Model: {self.boosted_tree_model}")
            self.logger.info(f"Output: {self.boosted_tree_predictions_table}")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as num_predictions,
                AVG(absolute_error) as mean_absolute_error,
                SQRT(AVG(POWER(error, 2))) as root_mean_squared_error,
                AVG(predicted_rating) as avg_predicted_rating,
                AVG(actual_rating) as avg_actual_rating
            FROM `{self.boosted_tree_predictions_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            self.logger.info("BOOSTED_TREE_REGRESSOR Predictions Generated:")
            self.logger.info(f"  Predictions: {stats['num_predictions'].iloc[0]}")
            self.logger.info(f"  MAE: {stats['mean_absolute_error'].iloc[0]:.4f}")
            self.logger.info(f"  RMSE: {stats['root_mean_squared_error'].iloc[0]:.4f}")
            self.logger.info(f"  Avg Predicted: {stats['avg_predicted_rating'].iloc[0]:.4f}")
            self.logger.info(f"  Avg Actual: {stats['avg_actual_rating'].iloc[0]:.4f}")

        except Exception as e:
            self.logger.error(f"Error generating BOOSTED_TREE_REGRESSOR predictions: {e}", exc_info=True)
            raise

    def create_combined_predictions(self):
        """
        Create a combined predictions table that merges all model predictions.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Creating Combined Predictions Table")
            self.logger.info("=" * 60)

            query = f"""
            CREATE OR REPLACE TABLE `{self.combined_predictions_table}` AS
            WITH automl_preds AS (
                SELECT
                    user_id_clean,
                    book_id,
                    actual_rating,
                    predicted_rating as automl_predicted_rating,
                    absolute_error as automl_absolute_error
                FROM `{self.automl_predictions_table}`
            ),
            boosted_tree_preds AS (
                SELECT
                    user_id_clean,
                    book_id,
                    predicted_rating as boosted_tree_predicted_rating,
                    absolute_error as boosted_tree_absolute_error
                FROM `{self.boosted_tree_predictions_table}`
            ),
            mf_recommendations AS (
                SELECT
                    user_id_clean,
                    book_id,
                    predicted_rating as mf_predicted_rating,
                    rank as mf_rank
                FROM `{self.mf_recommendations_table}`
            )
            SELECT
                COALESCE(a.user_id_clean, b.user_id_clean, m.user_id_clean) as user_id_clean,
                COALESCE(a.book_id, b.book_id, m.book_id) as book_id,
                a.actual_rating,
                a.automl_predicted_rating,
                a.automl_absolute_error,
                b.boosted_tree_predicted_rating,
                b.boosted_tree_absolute_error,
                m.mf_predicted_rating,
                m.mf_rank,
                -- Ensemble prediction (average of regression models)
                CASE 
                    WHEN a.automl_predicted_rating IS NOT NULL AND b.boosted_tree_predicted_rating IS NOT NULL
                    THEN (a.automl_predicted_rating + b.boosted_tree_predicted_rating) / 2.0
                    WHEN a.automl_predicted_rating IS NOT NULL
                    THEN a.automl_predicted_rating
                    WHEN b.boosted_tree_predicted_rating IS NOT NULL
                    THEN b.boosted_tree_predicted_rating
                    ELSE NULL
                END as ensemble_predicted_rating
            FROM automl_preds a
            FULL OUTER JOIN boosted_tree_preds b
                ON a.user_id_clean = b.user_id_clean AND a.book_id = b.book_id
            FULL OUTER JOIN mf_recommendations m
                ON COALESCE(a.user_id_clean, b.user_id_clean) = m.user_id_clean 
                AND COALESCE(a.book_id, b.book_id) = m.book_id
            """

            self.logger.info(f"Creating combined predictions table...")
            self.logger.info(f"Output: {self.combined_predictions_table}")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(DISTINCT user_id_clean) as num_users,
                COUNT(DISTINCT book_id) as num_books,
                COUNT(automl_predicted_rating) as automl_count,
                COUNT(boosted_tree_predicted_rating) as boosted_tree_count,
                COUNT(mf_predicted_rating) as mf_count,
                COUNT(ensemble_predicted_rating) as ensemble_count
            FROM `{self.combined_predictions_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            self.logger.info("Combined Predictions Table Created:")
            self.logger.info(f"  Total Predictions: {stats['total_predictions'].iloc[0]}")
            self.logger.info(f"  Users: {stats['num_users'].iloc[0]}")
            self.logger.info(f"  Books: {stats['num_books'].iloc[0]}")
            self.logger.info(f"  AUTOML Predictions: {stats['automl_count'].iloc[0]}")
            self.logger.info(f"  Boosted Tree Predictions: {stats['boosted_tree_count'].iloc[0]}")
            self.logger.info(f"  Matrix Factorization Recommendations: {stats['mf_count'].iloc[0]}")
            self.logger.info(f"  Ensemble Predictions: {stats['ensemble_count'].iloc[0]}")

        except Exception as e:
            self.logger.error(f"Error creating combined predictions: {e}", exc_info=True)
            raise

    def run(self, top_k=10):
        """
        Execute the complete prediction pipeline.
        
        Args:
            top_k (int): Number of top recommendations per user for matrix factorization
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("BigQuery ML Prediction Pipeline")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Generate predictions from all models
        self.generate_matrix_factorization_recommendations(top_k=top_k)
        self.generate_automl_predictions()
        self.generate_boosted_tree_predictions()
        
        # Create combined predictions table
        self.create_combined_predictions()

        # Log completion
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"All predictions generated successfully")
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)


def main():
    """Main function to run predictions."""
    predictor = BigQueryMLPredictions()
    predictor.run(top_k=10)


if __name__ == "__main__":
    main()

