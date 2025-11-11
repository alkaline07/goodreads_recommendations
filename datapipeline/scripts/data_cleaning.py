"""
Data Cleaning Module for Goodreads Recommendation System

This module handles the cleaning and preprocessing of raw Goodreads data from BigQuery.
It performs data validation, null handling, text cleaning, and creates author gender mappings.

Key Features:
- Cleans books and interactions tables from BigQuery
- Handles missing values with appropriate defaults
- Standardizes text fields and removes duplicates
- Creates author gender mapping using gender-guesser library
- Applies global median imputation for numeric columns

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datapipeline.scripts.logger_setup import get_logger
import time
from datetime import datetime
from gender_guesser.detector import Detector
from tqdm import tqdm
import sys

class DataCleaning:
    
    def __init__(self):
        """
        Initialize the DataCleaning class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for data cleaning operations
        - BigQuery client and project information
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        # Initialize logging for data cleaning operations
        self.logger = get_logger("data_cleaning")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id="books"
        

    def _clean_table_generic(self, table_name: str, destination_table: str, filter_clause: str = None):
        """
        Generic function to clean a BigQuery table by applying transformations.
        
        Args:
            dataset_id (str): BigQuery dataset ID containing the source table
            table_name (str): Name of the source table to clean
            destination_table (str): Full table ID for the cleaned destination table
            filter_clause (str, optional): A SQL WHERE clause to apply for filtering.
            
        The method performs the following cleaning operations:
        - Removes duplicates using SELECT DISTINCT
        - Handles null values with appropriate defaults
        - Cleans and standardizes text fields
        - Flattens array columns for easier processing
        """
        try:
            self.logger.info(f"Starting generic cleaning for table: {self.dataset_id}.{table_name}")

            # Get table schema information from BigQuery INFORMATION_SCHEMA
            # This allows us to dynamically handle different table structures
            columns_info = self.client.query(f"""
                SELECT column_name, data_type
                FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).to_dataframe(create_bqstorage_client=False)
            self.logger.info(f"Retrieved {len(columns_info)} columns for table {table_name}")

            # Categorize columns by data type for different cleaning strategies
            array_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'].startswith('ARRAY')]
            string_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'] in ('STRING', 'CHAR', 'TEXT')]
            bool_cols = [row['column_name'] for _, row in columns_info.iterrows() if row['data_type'] == 'BOOL']

            # Build SQL SELECT expressions for each column based on data type
            # Each column gets appropriate cleaning logic based on its type
            select_exprs = []
            for _, row in columns_info.iterrows():
                col = row['column_name']
                # For string columns: trim whitespace and replace empty strings with 'Unknown'
                if col in string_cols:
                    select_exprs.append(f"COALESCE(NULLIF(TRIM({col}), ''), 'Unknown') AS {col}_clean")
                # For boolean columns: replace NULL with FALSE
                elif col in bool_cols:
                    select_exprs.append(f"COALESCE({col}, FALSE) AS {col}")
                # For array columns: flatten and convert to JSON strings, filtering out NULLs
                elif col in array_cols:
                    select_exprs.append(
                        f"ARRAY(SELECT TO_JSON_STRING(x) FROM UNNEST({col}) AS x WHERE x IS NOT NULL) AS {col}_flat"
                    )
                # For other columns: keep as-is
                else:
                    select_exprs.append(col)

            # Join all SELECT expressions with proper formatting
            select_sql = ",\n  ".join(select_exprs)

            # Build the final SQL query based on whether a filter clause is provided
            if filter_clause:
                self.logger.info(f"Applying custom filter clause for {table_name}...")
                query = f"""
                SELECT DISTINCT
                {select_sql}
                FROM `{self.project_id}.{self.dataset_id}.{table_name}`
                {filter_clause}
                """
            else:
                self.logger.info(f"Skipping filter clause for {table_name}...")
                query = f"""
                SELECT DISTINCT
                {select_sql}
                FROM `{self.project_id}.{self.dataset_id}.{table_name}`
                """

            # Execute the cleaning query and save results to destination table
            self.logger.info(f"Executing cleaning query for {table_name}...")
            job_config = bigquery.QueryJobConfig(
                destination=destination_table,
                write_disposition="WRITE_TRUNCATE"  # Overwrite existing table if it exists
            )
            self.client.query(query, job_config=job_config).result()
            self.logger.info(f" Cleaned table saved: {destination_table}")

        except Exception as e:
            # Log any errors that occur during the cleaning process
            self.logger.error(f"Error cleaning table {self.dataset_id}.{table_name}: {e}", exc_info=True)

    def clean_books_table(self, books_table_name: str, books_destination: str):
        """
        Cleans the books table, applying specific outlier filters for pages and year.
        """
        self.logger.info(f"Cleaning books table: {self.dataset_id}.{books_table_name}")
        # Specific outlier filter for books table
        filter_clause = """
        WHERE (num_pages >= 10 AND num_pages <= 2000)
          AND (publication_year >= 1900 AND publication_year <= 2025)
        """
        self._clean_table_generic(
            table_name=books_table_name,
            destination_table=books_destination,
            filter_clause=filter_clause
        )

    def clean_interactions_table(self, interactions_table_name: str, interactions_destination: str, books_destination: str):
        """
        Cleans the interactions table (no custom outlier filtering).
        """
        self.logger.info(f"Cleaning interactions table: {self.dataset_id}.{interactions_table_name}")
        self._clean_table_generic(
            table_name=interactions_table_name,
            destination_table=interactions_destination,
            filter_clause=None # No filter
        )

        # --- Filter interactions table based on cleaned books ---
        try:
            self.logger.info(f"Filtering interactions table ({interactions_destination}) against books table ({books_destination})...")
            
            filter_query = f"""
            SELECT t1.*
            FROM `{interactions_destination}` AS t1
            INNER JOIN `{books_destination}` AS t2
            ON t1.book_id = t2.book_id
            """
            
            job_config = bigquery.QueryJobConfig(
                destination=interactions_destination, # Overwrite the interactions table
                write_disposition="WRITE_TRUNCATE"
            )
            
            query_job = self.client.query(filter_query, job_config=job_config)
            query_job.result()
            
            self.logger.info(f"Successfully filtered interactions table: {interactions_destination}")

        except Exception as e:
            self.logger.error(f"Error filtering interactions table: {e}", exc_info=True)

    def run(self):
        """
        Execute the complete data cleaning pipeline.
        
        This method orchestrates the cleaning of both books and interactions tables,
        applies appropriate cleaning strategies for each table type, and creates
        author gender mappings for bias analysis.
        """
        # Initialize pipeline execution with logging
        self.logger.info("=" * 60)
        self.logger.info("Good Reads Data Cleaning Pipeline")
        start_time = time.time()
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # Define destination table IDs
        books_table_id = f"{self.project_id}.books.goodreads_books_cleaned_staging"
        interactions_table_id = f"{self.project_id}.books.goodreads_interactions_cleaned_staging"
        
        # Clean books table with outlier filtering
        self.clean_books_table(
            books_table_name="goodreads_books_mystery_thriller_crime",
            books_destination=books_table_id
        )

        # Clean interactions table (initial clean, no outlier filtering)
        self.clean_interactions_table(
            interactions_table_name="goodreads_interactions_mystery_thriller_crime",
            interactions_destination=interactions_table_id,
            books_destination=books_table_id
        )

        # Fetch and log sample rows from cleaned tables for verification
        try:
            # Get sample data from cleaned books table
            df_books_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_books_cleaned_staging` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)

            # Get sample data from cleaned interactions table
            df_interactions_sample = self.client.query(
                f"SELECT * FROM `{self.project_id}.books.goodreads_interactions_cleaned_staging` LIMIT 5"
            ).to_dataframe(create_bqstorage_client=False)

            # Log sample data for verification
            self.logger.info("Books sample:")
            self.logger.info("\n%s", df_books_sample)
            self.logger.info("Interactions sample:")
            self.logger.info("\n%s", df_interactions_sample)
        except Exception as e:
            self.logger.error(f"Error fetching sample data: {e}", exc_info=True)
            print("Books sample:")
            
        # Create author gender mapping for bias analysis
        self.create_author_gender_map()
        
        # Log completion
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)

    def create_author_gender_map(self):
        """
        Generate and upload author gender mapping table to BigQuery.
        
        This method creates a gender mapping for authors to support bias analysis
        in the recommendation system. It uses the gender-guesser library to infer
        gender from author names and stores the results in BigQuery.
        
        The gender mapping is used later in the bias analysis pipeline to ensure
        fair recommendations across different author demographics.
        """
        try:
            self.logger.info("Starting gender mapping for authors...")

            # Load authors table from BigQuery using dynamic project ID
            query = f"""
                SELECT author_id, name
                FROM `{self.project_id}.books.goodreads_book_authors`
                WHERE name IS NOT NULL
            """
            authors_df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            self.logger.info(f"Retrieved {len(authors_df)} author rows.")

            # Initialize gender detector with case-insensitive matching
            detector = Detector(case_sensitive=False)

            def get_gender(name):
                """
                Infer gender from author name using gender-guesser library.
                
                Args:
                    name (str): Author's full name
                    
                Returns:
                    str: 'Male', 'Female', or 'Unknown'
                """
                # Handle edge cases: empty names, names with periods, or single characters
                if not name or '.' in name or len(name.split()) == 0:
                    return "Unknown"
                    
                # Use first name for gender inference
                g = detector.get_gender(name.split()[0])
                
                # Map gender-guesser results to our categories
                if g in ["male", "mostly_male"]:
                    return "Male"
                elif g in ["female", "mostly_female"]:
                    return "Female"
                else:
                    return "Unknown"

            tqdm.pandas(desc="Inferring author gender", file=sys.stdout)
            authors_df["author_gender_group"] = authors_df["name"].progress_apply(get_gender)

            # Upload gender mapping back to BigQuery
            table_id = f"{self.project_id}.books.goodreads_author_gender_map"
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            job = self.client.load_table_from_dataframe(authors_df, table_id, job_config=job_config)
            job.result()  # Wait for upload to complete
            self.logger.info(f"Uploaded {len(authors_df)} rows to {table_id}")
            self.logger.info("Uploaded gender map to books.goodreads_author_gender_map")

        except Exception as e:
            self.logger.error(f"Error creating author gender map: {e}", exc_info=True)


def main():
    """
    Main entry point for the data cleaning script.
    
    This function is called by the Airflow DAG to execute the data cleaning pipeline.
    It creates a DataCleaning instance and runs the complete cleaning process.
    """
    data_cleaner = DataCleaning()
    data_cleaner.run()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()