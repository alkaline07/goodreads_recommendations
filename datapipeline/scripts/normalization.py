"""
Data Normalization Module for Goodreads Recommendation System

This module applies normalization transformations to the features table to prepare
data for machine learning model training. It performs log transformations on
skewed features and user-centered rating normalization.

Key Transformations:
- Log transformation for skewed numeric features (popularity, activity counts, etc.)
- User-centered rating normalization to remove individual rating biases
- Handles edge cases for zero and negative values

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger
import time

class GoodreadsNormalization:

    def __init__(self):
        """
        Initialize the GoodreadsNormalization class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for normalization operations
        - BigQuery client and project information
        - Target table reference for normalization operations
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME", ".") + "/gcp_credentials.json"

        # Initialize logging for normalization operations
        self.logger = get_logger("normalization")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        
        # Target table for normalization operations (features table)
        self.table = f"{self.project_id}.{self.dataset_id}.goodreads_features_cleaned_staging"

    def log_transform_features(self):
        """
        Apply log transformations to skewed numeric features.
        
        This method applies natural log transformation to features that are highly
        skewed (e.g., popularity scores, activity counts) to make them more normally
        distributed for machine learning algorithms. It handles edge cases for
        zero and negative values appropriately.
        """
        try:
            self.logger.info("Applying log transformations to skewed features...")
            
            # Apply log transformations to various skewed features
            # Uses LN(x + 1) to handle zero values and CASE statements for edge cases
            query = f"""
            UPDATE `{self.table}`
            SET
              popularity_score = CAST(LN(popularity_score + 1) AS INT64),
              user_activity_count = CAST(LN(user_activity_count + 1) AS INT64),
              description_length = CAST(LN(description_length + 1) AS INT64),
              num_books_read = CASE
                  WHEN num_books_read > 0 THEN CAST(LN(num_books_read + 1) AS INT64)
                  ELSE num_books_read
              END,
              user_days_to_read = CASE
                  WHEN user_days_to_read > 0 THEN CAST(LN(user_days_to_read) AS INT64)
                  ELSE NULL
              END,
              ratings_count = CASE
                  WHEN ratings_count > 0 THEN CAST(LN(ratings_count + 1) AS INT64)
                  ELSE ratings_count
              END,
              num_pages = CASE
                  WHEN num_pages > 0 THEN CAST(LN(num_pages + 1) AS INT64)
                  ELSE num_pages
              END
            WHERE TRUE;
            """
            self.client.query(query).result()
            self.logger.info("Log transformations applied successfully.")
        except Exception as e:
            self.logger.error("Error applying log transformations.", exc_info=True)
            raise

    def normalize_user_ratings(self):
        """
        Apply user-centered rating normalization.
        
        This method normalizes ratings by subtracting each user's average rating
        from their individual ratings. This removes individual rating biases
        (e.g., some users rate higher/lower on average) and makes ratings more
        comparable across users for machine learning algorithms.
        """
        try:
            self.logger.info("Applying user-centered rating normalization...")
            
            # First, alter the rating column to FLOAT64 to handle decimal results
            alter_query = f"""
            ALTER TABLE `{self.table}`
            ALTER COLUMN rating SET DATA TYPE FLOAT64;
            """
            self.client.query(alter_query).result()

            # Apply user-centered normalization by subtracting user's average rating
            update_query = f"""
            UPDATE `{self.table}`
            SET rating = rating - avg_rating_given
            WHERE TRUE;
            """
            self.client.query(update_query).result()
            self.logger.info("User-centered rating normalization applied successfully.")
        except Exception as e:
            self.logger.error("Error normalizing ratings.", exc_info=True)
            raise

    def run(self):
        """
        Execute the complete normalization pipeline.
        
        This method orchestrates the normalization process:
        1. Applies log transformations to skewed features
        2. Performs user-centered rating normalization
        
        The pipeline prepares the features table for machine learning model training
        by ensuring features are properly scaled and normalized.
        """
        # Initialize pipeline execution with logging
        self.logger.info("="*60)
        self.logger.info("Good Reads Normalization Pipeline")
        start_time = time.time()
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Step 1: Apply log transformations to skewed features
        self.log_transform_features()

        # Step 2: Apply user-centered rating normalization
        self.normalize_user_ratings()

        # Log pipeline completion statistics
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)


def main():
    """
    Main entry point for the normalization script.
    
    This function is called by the Airflow DAG to execute the normalization pipeline.
    It creates a GoodreadsNormalization instance and runs the complete normalization process.
    """
    normalizer = GoodreadsNormalization()
    normalizer.run()

if __name__ == "__main__":
    main()
