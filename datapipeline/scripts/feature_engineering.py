"""
Feature Engineering Module for Goodreads Recommendation System

This module creates comprehensive machine learning features from cleaned Goodreads data.
It generates 40+ features across three levels: book-level, user-level, and interaction-level.

Key Features Generated:
- Book-level: popularity metrics, reading difficulty, content features, temporal features
- User-level: reading patterns, preferences, activity metrics
- Interaction-level: user-book specific features, temporal patterns

The module uses BigQuery for scalable feature computation and creates a final
features table ready for machine learning model training.

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger
import time

class FeatureEngineering:

    def __init__(self):
        """
        Initialize the FeatureEngineering class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for feature engineering operations
        - BigQuery client and project information
        - Source and destination table references
        - Feature engineering parameters and constraints
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME", ".") + "/gcp_credentials.json"

        # Initialize logging for feature engineering operations
        self.logger = get_logger("feature_engineering")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # Define source and destination table references
        self.books_table = f"{self.project_id}.{self.dataset_id}.goodreads_books_cleaned_staging"
        self.interactions_table = f"{self.project_id}.{self.dataset_id}.goodreads_interactions_cleaned_staging"
        self.destination_table = f"{self.project_id}.{self.dataset_id}.goodreads_features_cleaned_staging"

        # Feature engineering parameters and constraints
        self.MIN_READING_DAYS = 1      # Minimum valid reading time in days
        self.MAX_READING_DAYS = 365    # Maximum valid reading time in days
        self.DEFAULT_PAGE_COUNT = 300  # Default page count for books with missing data
        self.DEFAULT_READING_DAYS = 14 # Default reading time for books with missing data

    def create_features(self):
        """
        Create comprehensive machine learning features from cleaned data.
        
        This method builds a complex BigQuery SQL query that generates 40+ features
        across three levels:
        1. Book-level features: popularity, difficulty, content characteristics
        2. User-level features: reading patterns, preferences, activity metrics
        3. Interaction-level features: user-book specific patterns and relationships
        
        The query uses CTEs (Common Table Expressions) to organize the feature
        engineering logic and creates a final merged table with all features.

        Updates:
        - derived 'interaction_weight' from 'interaction_type' (5=read, 4=like, 3=add_to_list, 1=click)
        - derived 'is_read' flag from 'interaction_type'
        """
        try:
            self.logger.info(f"Starting feature engineering pipeline")
            self.logger.info(f"Source tables: {self.books_table}, {self.interactions_table}")
            self.logger.info(f"Destination: {self.destination_table}")

            # Build the comprehensive feature engineering query
            query = f"""
            -- ==============================================================
            -- BOOK-LEVEL FEATURES (including average reading time per book)
            -- ==============================================================
            WITH book_reading_times AS (
              -- Calculate average reading time for each book across all readers
              SELECT
                book_id,
                AVG(
                  CASE 
                    WHEN SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean) IS NOT NULL 
                     AND SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean) IS NOT NULL
                     AND DATE_DIFF(
                       DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                       DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                       DAY
                     ) BETWEEN {self.MIN_READING_DAYS} AND {self.MAX_READING_DAYS}
                    THEN DATE_DIFF(
                      DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                      DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                      DAY
                    )
                    ELSE NULL
                  END
                ) AS avg_book_reading_time,
                COUNT(
                  CASE 
                    WHEN SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean) IS NOT NULL 
                     AND SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean) IS NOT NULL
                    THEN 1 
                  END
                ) AS num_readers_with_time
              FROM `{self.interactions_table}`
              WHERE is_read = TRUE
              GROUP BY book_id
            ),

            base_books AS (
              SELECT
                b.book_id,
                COALESCE(b.title_clean, '') AS title_clean,
                COALESCE(b.average_rating, 0.0) AS average_rating,
                COALESCE(b.ratings_count, 0) AS ratings_count,
                COALESCE(b.text_reviews_count, 0) AS text_reviews_count,
                LOG(1 + COALESCE(b.ratings_count, 0)) AS log_ratings_count,
                COALESCE(b.ratings_count, 0) + COALESCE(b.text_reviews_count, 0) AS popularity_score,
                LENGTH(COALESCE(b.title_clean, '')) AS title_length_in_characters,
                ARRAY_LENGTH(SPLIT(COALESCE(b.title_clean, ''), ' ')) AS title_length_in_words,
                LENGTH(COALESCE(b.description_clean, '')) AS description_length,
                COALESCE(ARRAY_LENGTH(b.popular_shelves_flat), 0) AS num_genres,
                COALESCE(ARRAY_LENGTH(b.series_flat) > 0, FALSE) AS is_series,

                -- Pages and publication year features
                COALESCE(b.num_pages, {self.DEFAULT_PAGE_COUNT}) AS num_pages,
                COALESCE(b.publication_year, EXTRACT(YEAR FROM CURRENT_DATE()) - 5) AS publication_year,

                -- Book age calculation
                EXTRACT(YEAR FROM CURRENT_DATE()) - COALESCE(b.publication_year, EXTRACT(YEAR FROM CURRENT_DATE()) - 5) AS book_age_years,

                -- Book-level average reading time
                COALESCE(rt.avg_book_reading_time, {self.DEFAULT_READING_DAYS}) AS avg_book_reading_time_days,
                COALESCE(rt.num_readers_with_time, 0) AS num_readers_with_reading_time,

                -- Adjusted rating (Bayesian average)
                CASE
                  WHEN COALESCE(b.ratings_count, 0) = 0 THEN 3.0
                  ELSE (
                    1 + 4 * (
                      ((COALESCE(b.average_rating, 3.0) - 1)/4) + POW(1.96,2)/(2*b.ratings_count)
                      - 1.96 * SQRT((( (COALESCE(b.average_rating, 3.0) - 1)/4)*(1 - (COALESCE(b.average_rating, 3.0) - 1)/4) + POW(1.96,2)/(4*b.ratings_count)) / b.ratings_count)
                    ) / (1 + POW(1.96,2)/b.ratings_count)
                  )
                END AS adjusted_average_rating
              FROM `{self.books_table}` b
              LEFT JOIN book_reading_times rt ON b.book_id = rt.book_id
              WHERE b.book_id IS NOT NULL
            ),

            threshold AS (
              SELECT 
                APPROX_QUANTILES(adjusted_average_rating, 100)[OFFSET(80)] AS rating_threshold
              FROM base_books
            ),

            book_final AS (
              SELECT
                b.*,
                COALESCE(b.adjusted_average_rating >= t.rating_threshold, FALSE) AS great,
                COALESCE(SAFE_DIVIDE(
                  b.popularity_score - MIN(b.popularity_score) OVER(),
                  NULLIF(MAX(b.popularity_score) OVER() - MIN(b.popularity_score) OVER(), 0)
                ), 0.5) AS book_popularity_normalized,

                -- Book difficulty indicator based on reading time
                CASE 
                  WHEN avg_book_reading_time_days <= 7 THEN 'quick_read'
                  WHEN avg_book_reading_time_days <= 14 THEN 'moderate'
                  WHEN avg_book_reading_time_days <= 30 THEN 'long_read'
                  ELSE 'very_long'
                END AS reading_pace_category,

                -- Pages-based categories
                CASE 
                  WHEN num_pages <= 200 THEN 'short'
                  WHEN num_pages <= 350 THEN 'medium'
                  WHEN num_pages <= 500 THEN 'long'
                  ELSE 'very_long'
                END AS book_length_category,

                -- Average pages per day (for books with reading time data)
                COALESCE(
                  SAFE_DIVIDE(num_pages, NULLIF(avg_book_reading_time_days, 0)),
                  25
                ) AS avg_pages_per_day,

                -- Era classification
                CASE 
                  WHEN book_age_years <= 2 THEN 'new_release'
                  WHEN book_age_years <= 5 THEN 'recent'
                  WHEN book_age_years <= 10 THEN 'contemporary'
                  WHEN book_age_years <= 20 THEN 'modern'
                  ELSE 'classic'
                END AS book_era
              FROM base_books b CROSS JOIN threshold t
            ),

            -- ==============================================================
            -- USER-LEVEL FEATURES
            -- ==============================================================
            user_features AS (
              SELECT
                user_id_clean,
                COALESCE(COUNTIF(is_read), 0) AS num_books_read,
                COALESCE(AVG(IF(rating > 0, rating, NULL)), 3.0) AS avg_rating_given,
                COUNT(book_id) AS user_activity_count,
                COALESCE(
                  DATE_DIFF(
                    CURRENT_DATE(),
                    DATE(MAX(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', date_updated_clean))),
                    DAY
                  ), 
                  365
                ) AS recent_activity_days,
                -- User's personal average reading speed
                COALESCE(
                  AVG(
                    CASE 
                      WHEN SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean) IS NOT NULL 
                       AND SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean) IS NOT NULL
                       AND DATE_DIFF(
                         DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                         DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                         DAY
                       ) BETWEEN {self.MIN_READING_DAYS} AND {self.MAX_READING_DAYS}
                      THEN DATE_DIFF(
                        DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                        DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                        DAY
                      )
                      ELSE NULL
                    END
                  ), 
                  {self.DEFAULT_READING_DAYS}
                ) AS user_avg_reading_time_days
              FROM `{self.interactions_table}`
              WHERE user_id_clean IS NOT NULL
              GROUP BY user_id_clean
            ),

            -- ==============================================================
            -- INTERACTION-LEVEL FEATURES
            -- ==============================================================
            interaction_features AS (
              SELECT
                user_id_clean,
                book_id,
                COALESCE(rating, 0) AS rating,
                COALESCE(is_read, FALSE) AS is_read,

                -- Feature: Explicit Interaction Score
                -- Based on the hierarchy: read > like > add_to_list > click
                CASE 
                    WHEN interaction_type = 'read' THEN 5
                    WHEN interaction_type = 'like' THEN 4
                    WHEN interaction_type = 'add_to_list' THEN 3
                    WHEN interaction_type = 'click' THEN 1
                    ELSE 0
                END as interaction_weight,

                COALESCE(
                  CASE 
                    WHEN SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean) IS NOT NULL 
                     AND SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean) IS NOT NULL
                     AND DATE_DIFF(
                       DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                       DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                       DAY
                     ) BETWEEN {self.MIN_READING_DAYS} AND {self.MAX_READING_DAYS}
                    THEN DATE_DIFF(
                      DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                      DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', started_at_clean)),
                      DAY
                    )
                    ELSE NULL
                  END,
                  {self.DEFAULT_READING_DAYS}
                ) AS user_days_to_read,
                COALESCE(
                  DATE_DIFF(
                    CURRENT_DATE(),
                    DATE(SAFE.PARSE_TIMESTAMP('%a %b %d %H:%M:%S %z %Y', read_at_clean)),
                    DAY
                  ),
                  365
                ) AS user_book_recency
              FROM `{self.interactions_table}`
              WHERE user_id_clean IS NOT NULL AND book_id IS NOT NULL
            ),

            -- ==============================================================
            -- FINAL MERGE
            -- ==============================================================
            merged AS (
              SELECT
                i.*,
                u.num_books_read,
                u.avg_rating_given,
                u.user_activity_count,
                u.recent_activity_days,
                u.user_avg_reading_time_days,
                b.title_clean,
                b.average_rating,
                b.adjusted_average_rating,
                b.great,
                b.ratings_count,
                b.log_ratings_count,
                b.popularity_score,
                b.book_popularity_normalized,
                b.num_genres,
                b.is_series,
                b.title_length_in_characters,
                b.title_length_in_words,
                b.description_length,

                -- Pages and publication features
                b.num_pages,
                b.publication_year,
                b.book_age_years,
                b.book_length_category,
                b.book_era,
                b.avg_pages_per_day,

                -- Reading time features
                b.avg_book_reading_time_days,
                b.num_readers_with_reading_time,
                b.reading_pace_category,

                COALESCE(u.avg_rating_given - b.adjusted_average_rating, 0.0) AS user_avg_rating_vs_book,

                -- Compare user's reading speed to average for this book
                COALESCE(
                  SAFE_DIVIDE(i.user_days_to_read, NULLIF(b.avg_book_reading_time_days, 0)),
                  1.0
                ) AS user_reading_speed_ratio,

                -- User's pages per day for this specific book
                COALESCE(
                  SAFE_DIVIDE(b.num_pages, NULLIF(i.user_days_to_read, 0)),
                  25
                ) AS user_pages_per_day_this_book
              FROM interaction_features i
              INNER JOIN user_features u ON i.user_id_clean = u.user_id_clean
              INNER JOIN book_final b ON i.book_id = b.book_id
            )

            SELECT * FROM merged
            WHERE 
              num_books_read > 0
              AND ratings_count > 0
              AND num_pages > 0
              AND publication_year > 1900
              AND publication_year <= EXTRACT(YEAR FROM CURRENT_DATE())
            """

            # Execute the feature engineering query and save results
            job_config = bigquery.QueryJobConfig(
                destination=self.destination_table,
                write_disposition="WRITE_TRUNCATE"  # Overwrite existing table if it exists
            )

            self.logger.info("Executing feature engineering query...")
            query_job = self.client.query(query, job_config=job_config)
            query_job.result()  # Wait for query completion
            self.logger.info(f" Features table successfully created: {self.destination_table}")

        except Exception as e:
            # Log any errors that occur during feature engineering
            self.logger.error(f"Error in feature engineering: {e}", exc_info=True)
            raise

    def get_table_stats(self):
        """
        Gather and log comprehensive statistics about the generated features table.
        
        This method queries the features table to get key metrics and performs
        basic anomaly detection to ensure data quality. It logs statistics
        about row counts, user counts, book counts, and average values.
        """
        try:
            self.logger.info("Gathering table statistics...")
            
            # Query to get comprehensive statistics about the features table
            stats_query = f"""
            SELECT 
              COUNT(*) as total_rows,
              COUNT(DISTINCT user_id_clean) as unique_users,
              COUNT(DISTINCT book_id) as unique_books,
              ROUND(AVG(num_books_read), 2) as avg_books_per_user,
              ROUND(AVG(avg_book_reading_time_days), 2) as avg_reading_time_days,
              ROUND(AVG(num_pages), 0) as avg_pages,
              ROUND(AVG(rating), 2) as avg_rating
            FROM `{self.destination_table}`
            """

            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            # Log comprehensive statistics
            self.logger.info("Table Statistics:")
            self.logger.info(f"Total rows: {stats['total_rows'].iloc[0]:,}")
            self.logger.info(f"Unique users: {stats['unique_users'].iloc[0]:,}")
            self.logger.info(f"Unique books: {stats['unique_books'].iloc[0]:,}")
            self.logger.info(f"Avg books per user: {stats['avg_books_per_user'].iloc[0]}")
            self.logger.info(f"Avg reading time (days): {stats['avg_reading_time_days'].iloc[0]}")
            self.logger.info(f"Avg pages: {stats['avg_pages'].iloc[0]}")
            self.logger.info(f"Avg rating: {stats['avg_rating'].iloc[0]}")

            # Perform basic data anomaly detection
            if stats['total_rows'].iloc[0] == 0:
                self.logger.warning(
                    "Anomaly Detected: No rows found in the final features table!")

            if stats['avg_rating'].iloc[0] < 1.0 or stats['avg_rating'].iloc[0] > 5.0:
                self.logger.warning(
                    "Anomaly Detected: Average rating is outside the expected range (1–5).")

            if stats['unique_users'].iloc[0] < 1:
                self.logger.warning(
                    "Anomaly Detected: Very few unique users found — potential data loss.")

        except Exception as e:
            self.logger.error(f"Error getting table stats: {e}", exc_info=True)

    def export_sample(self, sample_size=1000):
        """
        Export a sample of the features table for analysis and verification.
        
        Args:
            sample_size (int): Number of rows to sample from the features table
            
        This method creates a random sample of the features table and saves it
        as a Parquet file for analysis, debugging, or model development.
        """
        try:
            self.logger.info(f"Exporting sample of {sample_size} rows...")

            # Use BigQuery's TABLESAMPLE to get a random sample efficiently
            sample_query = f"""
            SELECT * 
            FROM `{self.destination_table}` 
            TABLESAMPLE SYSTEM (0.1 PERCENT)
            LIMIT {sample_size}
            """

            sample_df = self.client.query(sample_query).to_dataframe(create_bqstorage_client=False)

            # Create data directory if it doesn't exist
            os.makedirs("data/sample_features", exist_ok=True)

            # Save sample as Parquet file with timestamp
            output_path = f"data/sample_features/features_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            sample_df.to_parquet(output_path, index=False)

            self.logger.info(f" Sample saved to {output_path}")
            self.logger.info(f"   Shape: {sample_df.shape}")

            # Display sample data preview for verification
            self.logger.info("Sample data preview:")
            display_cols = ['user_id_clean', 'book_id', 'rating', 'num_pages', 'book_era']
            if all(col in sample_df.columns for col in display_cols):
                self.logger.info("\n%s", sample_df[display_cols].head())

        except Exception as e:
            self.logger.error(f"Error exporting sample: {e}", exc_info=True)

    def run(self):
        """
        Execute the complete feature engineering pipeline.
        
        This method orchestrates the entire feature engineering process:
        1. Creates comprehensive features from cleaned data
        2. Gathers and logs table statistics
        3. Exports a sample for analysis and verification
        
        The pipeline generates 40+ features across book, user, and interaction levels
        and creates a final features table ready for machine learning model training.
        """
        # Initialize pipeline execution with logging
        self.logger.info("=" * 60)
        self.logger.info("Good Reads Feature Engineering Pipeline")
        start_time = time.time()
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Step 1: Create comprehensive features from cleaned data
        self.create_features()

        # Step 2: Gather and log statistics about the generated features
        self.get_table_stats()

        # Step 3: Export a sample for analysis and verification
        self.export_sample(sample_size=1000)
        
        # Log pipeline completion statistics
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)


def main():
    """
    Main entry point for the feature engineering script.
    
    This function is called by the Airflow DAG to execute the feature engineering pipeline.
    It creates a FeatureEngineering instance and runs the complete feature creation process.
    """
    feature_engineer = FeatureEngineering()
    feature_engineer.run()


if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()