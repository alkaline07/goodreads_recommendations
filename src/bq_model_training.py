"""
BigQuery ML Model Training Module for Goodreads Recommendation System

This module trains three BigQuery ML models for the recommendation system:
1. MATRIX_FACTORIZATION - For collaborative filtering recommendations
2. AUTOML_REGRESSOR - For automated rating prediction
3. BOOSTED_TREE_REGRESSOR - For feature-rich rating prediction

The models are trained on the training dataset and evaluated on validation data.

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger
import time


class BigQueryMLModelTraining:
    """
    Class for training BigQuery ML models for recommendations and rating prediction.
    """

    def __init__(self):
        """
        Initialize the BigQuery ML Model Training class.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration
        - BigQuery client and project information
        - Training, validation, and model table references
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
        self.logger = get_logger("model_training")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # Define table references
        self.train_table = f"{self.project_id}.{self.dataset_id}.goodreads_train_set"
        self.val_table = f"{self.project_id}.{self.dataset_id}.goodreads_validation_set"
        self.test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"

        # Model table references
        self.matrix_factorization_model = f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
        self.automl_regressor_model = f"{self.project_id}.{self.dataset_id}.automl_regressor_model"
        self.boosted_tree_model = f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"

    def get_feature_columns(self):
        """
        Get the list of feature columns to use for model training.
        
        Returns:
            list: List of feature column names
        """
        # Get schema from training table to identify feature columns
        try:
            table = self.client.get_table(self.train_table)
            all_columns = [field.name for field in table.schema]
            
            # Exclude non-feature columns
            exclude_columns = {
                'user_id_clean', 'book_id', 'rating', 'is_read',
                'user_days_to_read', 'user_book_recency'  # These are interaction-level, may want to exclude
            }
            
            feature_columns = [col for col in all_columns if col not in exclude_columns]
            self.logger.info(f"Found {len(feature_columns)} feature columns")
            return feature_columns
        except Exception as e:
            self.logger.error(f"Error getting feature columns: {e}", exc_info=True)
            # Return default feature columns if schema lookup fails
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

    def train_matrix_factorization(self):
        """
        Train MATRIX_FACTORIZATION model for collaborative filtering recommendations.
        
        This model learns user-item interactions and can generate recommendations
        based on collaborative filtering patterns.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Training MATRIX_FACTORIZATION Model")
            self.logger.info("=" * 60)

            # MATRIX_FACTORIZATION requires user_id, item_id, and optionally rating
            # We'll use rating as the feedback signal
            query = f"""
            CREATE OR REPLACE MODEL `{self.matrix_factorization_model}`
            OPTIONS(
                model_type='MATRIX_FACTORIZATION',
                user_col='user_id_clean',
                item_col='book_id',
                rating_col='rating',
                feedback_type='EXPLICIT',  -- Use explicit feedback (ratings)
                l2_reg=0.1,
                num_factors=10,
                max_iterations=15
            ) AS
            SELECT
                user_id_clean,
                book_id,
                rating
            FROM `{self.train_table}`
            WHERE rating > 0  -- Only use explicit ratings
            """

            self.logger.info(f"Training MATRIX_FACTORIZATION model...")
            self.logger.info(f"Source: {self.train_table}")
            self.logger.info(f"Model: {self.matrix_factorization_model}")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            self.logger.info("MATRIX_FACTORIZATION model training completed successfully")

            # Get model evaluation metrics
            self.evaluate_model(self.matrix_factorization_model, "MATRIX_FACTORIZATION")

        except Exception as e:
            self.logger.error(f"Error training MATRIX_FACTORIZATION model: {e}", exc_info=True)
            raise

    def train_automl_regressor(self):
        """
        Train AUTOML_REGRESSOR model for automated rating prediction.
        
        This model automatically selects features and hyperparameters for rating prediction.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Training AUTOML_REGRESSOR Model")
            self.logger.info("=" * 60)

            feature_columns = self.get_feature_columns()
            feature_list = ", ".join(feature_columns)

            query = f"""
            CREATE OR REPLACE MODEL `{self.automl_regressor_model}`
            OPTIONS(
                model_type='AUTOML_REGRESSOR',
                input_label_cols=['rating'],
                budget_hours=1.0
            ) AS
            SELECT
                {feature_list},
                rating
            FROM `{self.train_table}`
            WHERE rating IS NOT NULL
            """

            self.logger.info(f"Training AUTOML_REGRESSOR model...")
            self.logger.info(f"Source: {self.train_table}")
            self.logger.info(f"Model: {self.automl_regressor_model}")
            self.logger.info(f"Using {len(feature_columns)} features")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            self.logger.info("AUTOML_REGRESSOR model training completed successfully")

            # Get model evaluation metrics
            self.evaluate_model(self.automl_regressor_model, "AUTOML_REGRESSOR")

        except Exception as e:
            self.logger.error(f"Error training AUTOML_REGRESSOR model: {e}", exc_info=True)
            raise

    def train_boosted_tree_regressor(self):
        """
        Train BOOSTED_TREE_REGRESSOR model for feature-rich rating prediction.
        
        This model uses gradient boosting for rating prediction with explicit feature selection.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Training BOOSTED_TREE_REGRESSOR Model")
            self.logger.info("=" * 60)

            feature_columns = self.get_feature_columns()
            feature_list = ", ".join(feature_columns)

            query = f"""
            CREATE OR REPLACE MODEL `{self.boosted_tree_model}`
            OPTIONS(
                model_type='BOOSTED_TREE_REGRESSOR',
                input_label_cols=['rating'],
                num_parallel_tree=10,
                max_tree_depth=6,
                min_split_loss=0.1,
                learning_rate=0.1,
                l1_reg=0.1,
                l2_reg=0.1,
                early_stop=True
            ) AS
            SELECT
                {feature_list},
                rating
            FROM `{self.train_table}`
            WHERE rating IS NOT NULL
            """

            self.logger.info(f"Training BOOSTED_TREE_REGRESSOR model...")
            self.logger.info(f"Source: {self.train_table}")
            self.logger.info(f"Model: {self.boosted_tree_model}")
            self.logger.info(f"Using {len(feature_columns)} features")

            job = self.client.query(query)
            job.result()  # Wait for job to complete

            self.logger.info("BOOSTED_TREE_REGRESSOR model training completed successfully")

            # Get model evaluation metrics
            self.evaluate_model(self.boosted_tree_model, "BOOSTED_TREE_REGRESSOR")

        except Exception as e:
            self.logger.error(f"Error training BOOSTED_TREE_REGRESSOR model: {e}", exc_info=True)
            raise

    def evaluate_model(self, model_name, model_type):
        """
        Evaluate a trained model on validation data.
        
        Args:
            model_name (str): Full path to the BigQuery ML model
            model_type (str): Type of model (for logging)
        """
        try:
            self.logger.info(f"Evaluating {model_type} model on validation data...")

            # Get evaluation metrics
            eval_query = f"""
            SELECT *
            FROM ML.EVALUATE(MODEL `{model_name}`, (
                SELECT *
                FROM `{self.val_table}`
                WHERE rating IS NOT NULL
            ))
            """

            eval_result = self.client.query(eval_query).to_dataframe(create_bqstorage_client=False)

            if not eval_result.empty:
                self.logger.info(f"{model_type} Model Evaluation Metrics:")
                for col in eval_result.columns:
                    value = eval_result[col].iloc[0]
                    self.logger.info(f"  {col}: {value}")

        except Exception as e:
            self.logger.warning(f"Error evaluating {model_type} model: {e}")
            # Don't raise - evaluation failure shouldn't stop the pipeline

    def run(self):
        """
        Execute the complete model training pipeline.
        
        Trains all three BigQuery ML models:
        1. MATRIX_FACTORIZATION for recommendations
        2. AUTOML_REGRESSOR for rating prediction
        3. BOOSTED_TREE_REGRESSOR for feature-rich rating prediction
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("BigQuery ML Model Training Pipeline")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Train all three models
        self.train_matrix_factorization()
        self.train_automl_regressor()
        self.train_boosted_tree_regressor()

        # Log completion
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"All models trained successfully")
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)


def main():
    """Main function to run model training."""
    trainer = BigQueryMLModelTraining()
    trainer.run()


if __name__ == "__main__":
    main()

