"""
Anomaly Detection and Data Quality Validation Module for Goodreads Data Pipeline

This module performs comprehensive data quality validation using BigQuery SQL queries.
It implements a two-stage validation system: pre-cleaning and post-cleaning validation
to ensure data integrity throughout the entire processing workflow.

Key Features:
- Pre-cleaning validation: Validates source data before any processing
- Post-cleaning validation: Validates cleaned data after processing
- Zero tolerance policy: Any violations stop the pipeline
- Email notifications: Automatic failure alerts to stakeholders
- BigQuery integration: Scalable validation using SQL queries

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import pandas as pd
from google.cloud import bigquery
from airflow.utils.email import send_email
from datapipeline.scripts.logger_setup import get_logger
import time
from datetime import datetime

class AnomalyDetection:
    
    def __init__(self):
        """
        Initialize the AnomalyDetection class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for anomaly detection operations
        - BigQuery client and project information
        - Dataset reference for validation queries
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        # Initialize logging for anomaly detection operations
        self.logger = get_logger("anomaly_detection")
        
        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset = "books"

    def validate_data_quality(self, use_cleaned_tables=False):
        """
        Perform comprehensive data quality validation using BigQuery SQL queries.
        
        Args:
            use_cleaned_tables (bool): If True, validate cleaned tables; if False, validate source tables
            
        Returns:
            bool: True if all validations pass, False otherwise
            
        This method orchestrates validation of both books and interactions tables
        and implements a zero-tolerance policy - any violations stop the pipeline.
        """
        try:
            validation_type = "cleaned" if use_cleaned_tables else "source"
            self.logger.info(f"Starting BigQuery data validation for {validation_type} tables...")
            
            # Validate books table using appropriate validation rules
            books_success = self.validate_books_with_bigquery(use_cleaned_tables)
            
            # Validate interactions table using appropriate validation rules
            interactions_success = self.validate_interactions_with_bigquery(use_cleaned_tables)
            
            # Check overall success - zero tolerance policy
            if not books_success or not interactions_success:
                self.send_failure_email(f"Data validation failed for {validation_type} tables - check logs for details")
                raise Exception(f"Data validation failed for {validation_type} tables - critical issues found")
            
            self.logger.info(f"All {validation_type} data quality validations passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise

    def get_table_structure(self, table_name):
        """
        Get table structure information from BigQuery INFORMATION_SCHEMA.
        
        Args:
            table_name (str): Name of the table to get structure for
            
        Returns:
            pandas.DataFrame: DataFrame with column information, or None if error
        """
        try:
            # Query BigQuery INFORMATION_SCHEMA to get column details
            columns_info = self.client.query(f"""
                SELECT column_name, data_type
                FROM `{self.project_id}.{self.dataset}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).to_dataframe(create_bqstorage_client=False)
            
            self.logger.info(f"Retrieved {len(columns_info)} columns for table {table_name}")
            return columns_info
        except Exception as e:
            self.logger.error(f"Error fetching table structure for {table_name}: {e}")
            return None

    def validate_books_with_bigquery(self, use_cleaned_tables=False):
        """
        Validate books table using BigQuery SQL queries with appropriate validation rules.
        
        Args:
            use_cleaned_tables (bool): If True, validate cleaned table; if False, validate source table
            
        Returns:
            bool: True if all validations pass, False otherwise
            
        This method applies different validation rules based on whether it's validating
        source data (pre-cleaning) or cleaned data (post-cleaning).
        """
        try:
            # Choose table based on validation type
            table_name = "goodreads_books_cleaned_staging" if use_cleaned_tables else "goodreads_books_mystery_thriller_crime"
            
            self.logger.info(f"Validating books table: {table_name}")
            
            # Check if table exists and get row count
            count_query = f"""
            SELECT COUNT(*) as row_count
            FROM `{self.project_id}.{self.dataset}.{table_name}`
            """
            count_result = self.client.query(count_query).to_dataframe(create_bqstorage_client=False)
            row_count = count_result['row_count'].iloc[0]
            
            if row_count == 0:
                self.logger.error("Books table is empty")
                return False
            
            self.logger.info(f"Books table has {row_count} rows")
            
            # Data quality validation queries
            if not use_cleaned_tables:
                validation_queries = [
                    {
                        "name": "Check for null book_id",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE book_id IS NULL
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check for null title",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE title IS NULL
                        """,
                        "max_allowed": 0
                    },
                ]
            else:
                validation_queries = [
                    {
                        "name": "Check for null title",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE title_clean IS NULL
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check publication_year range",
                        "query": f"""
                        SELECT COUNT(*) as invalid_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE publication_year IS NULL OR publication_year < 1900 OR publication_year > 2025
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check num_pages range",
                        "query": f"""
                        SELECT COUNT(*) as invalid_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE num_pages IS NULL OR num_pages < 10 OR num_pages > 2000
                        """,
                        "max_allowed": 0
                    }
                ]
            
            # Run validation queries (uniform handling for pre/post)
            all_passed = True
            for validation in validation_queries:
                try:
                    result = self.client.query(validation["query"]).to_dataframe(create_bqstorage_client=False)
                    invalid_count = result.iloc[0, 0]
                    if invalid_count > validation["max_allowed"]:
                        self.logger.error(f"{validation['name']}: {invalid_count} violations found (max allowed: {validation['max_allowed']})")
                        all_passed = False
                    else:
                        self.logger.info(f"{validation['name']}: PASSED ({invalid_count} violations)")
                except Exception as e:
                    self.logger.error(f"Error running validation '{validation['name']}': {e}")
                    all_passed = False
            
            if all_passed:
                self.logger.info("Books table validation passed")
                return True
            else:
                self.logger.error("Books table validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating books table: {e}")
            return False

    def validate_interactions_with_bigquery(self, use_cleaned_tables=False):
        """
        Validate interactions table using BigQuery SQL queries
        Args:
            use_cleaned_tables (bool): If True, validate cleaned table; if False, validate source table
        """
        try:
            # Choose table based on validation type
            table_name = "goodreads_interactions_cleaned_staging" if use_cleaned_tables else "goodreads_interactions_mystery_thriller_crime"
            
            self.logger.info(f"Validating interactions table: {table_name}")
            
            # Check if table exists and get row count
            count_query = f"""
            SELECT COUNT(*) as row_count
            FROM `{self.project_id}.{self.dataset}.{table_name}`
            """
            count_result = self.client.query(count_query).to_dataframe(create_bqstorage_client=False)
            row_count = count_result['row_count'].iloc[0]
            
            if row_count == 0:
                self.logger.error("Interactions table is empty")
                return False
            
            self.logger.info(f"Interactions table has {row_count} rows")
            
            # Data quality validation queries
            if not use_cleaned_tables:
                validation_queries = [
                    {
                        "name": "Check for null user_id",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE user_id IS NULL
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check for null book_id",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE book_id IS NULL
                        """,
                        "max_allowed": 0
                    },
                ]
            else:
                validation_queries = [
                    {
                        "name": "Check for null user_id",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE user_id_clean IS NULL
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check for null book_id",
                        "query": f"""
                        SELECT COUNT(*) as null_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE book_id IS NULL
                        """,
                        "max_allowed": 0
                    },
                    {
                        "name": "Check rating range",
                        "query": f"""
                        SELECT COUNT(*) as invalid_count
                        FROM `{self.project_id}.{self.dataset}.{table_name}`
                        WHERE rating < 0 OR rating > 5
                        """,
                        "max_allowed": 0
                    }
                ]
            
            # Run validation queries (uniform handling for pre/post)
            all_passed = True
            for validation in validation_queries:
                try:
                    result = self.client.query(validation["query"]).to_dataframe(create_bqstorage_client=False)
                    invalid_count = result.iloc[0, 0]
                    if invalid_count > validation["max_allowed"]:
                        self.logger.error(f"{validation['name']}: {invalid_count} violations found (max allowed: {validation['max_allowed']})")
                        all_passed = False
                    else:
                        self.logger.info(f"{validation['name']}: PASSED ({invalid_count} violations)")
                except Exception as e:
                    self.logger.error(f"Error running validation '{validation['name']}': {e}")
                    all_passed = False
            
            if all_passed:
                self.logger.info("Interactions table validation passed")
                return True
            else:
                self.logger.error("Interactions table validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating interactions table: {e}")
            return False

    

    def send_failure_email(self, message):
        """
        Send email notification for validation failures to alert stakeholders.
        
        Args:
            message (str): Error message describing the validation failure
            
        This method sends an HTML email notification when data validation fails,
        providing details about the failure and required actions.
        """
        try:
            subject = "[CRITICAL] Data Validation Failed - Goodreads Pipeline"
            
            # Create HTML email content with failure details
            html_content = f"""
            <h2>Data Validation Failure</h2>
            <p><strong>Pipeline:</strong> Goodreads Recommendation System</p>
            <p><strong>Status:</strong> FAILED - Pipeline stopped</p>
            <p><strong>Error:</strong> {message}</p>
            
            <p><strong>Action Required:</strong> Please investigate and fix the data quality issues before re-running the pipeline.</p>
            <p><em>This is an automated alert from the Goodreads Data Pipeline.</em></p>
            """
            
            # Send email notification to configured recipient
            send_email(to=os.environ.get("AIRFLOW__SMTP__SMTP_USER"), subject=subject, html_content=html_content)
            
            self.logger.info("Validation failure email sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

    def run_pre_validation(self):
        """
        Execute pre-cleaning data validation pipeline.
        
        This method validates source data before any cleaning operations to ensure
        basic data integrity and identify critical issues early in the pipeline.
        
        Returns:
            bool: True if validation passes, raises exception if it fails
        """
        try:
            # Initialize pre-cleaning validation with logging
            self.logger.info("=" * 60)
            self.logger.info("Pre-Cleaning Data Validation Pipeline")
            start_time = time.time()
            self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 60)
            
            # Validate source tables before any processing
            result = self.validate_data_quality(use_cleaned_tables=False)
            
            # Log completion statistics
            end_time = time.time()
            self.logger.info("=" * 60)
            self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
            self.logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in pre-cleaning validation: {e}")
            raise

    def run_post_validation(self):
        """
        Execute post-cleaning data validation pipeline.
        
        This method validates cleaned data after processing to ensure the cleaning
        operations were successful and the data meets quality standards for ML training.
        
        Returns:
            bool: True if validation passes, raises exception if it fails
        """
        try:
            # Initialize post-cleaning validation with logging
            self.logger.info("=" * 60)
            self.logger.info("Post-Cleaning Data Validation Pipeline")
            start_time = time.time()
            self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 60)
            
            # Validate cleaned tables after processing
            result = self.validate_data_quality(use_cleaned_tables=True)
            
            # Log completion statistics
            end_time = time.time()
            self.logger.info("=" * 60)
            self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
            self.logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in post-cleaning validation: {e}")
            raise

def main_pre_validation():
    """
    Pre-cleaning validation function - validates source tables.
    
    This function is called by the Airflow DAG to validate source data
    before any cleaning operations begin.
    
    Returns:
        bool: True if validation passes, raises exception if it fails
    """
    anomaly_detector = AnomalyDetection()
    return anomaly_detector.run_pre_validation()

def main_post_validation():
    """
    Post-cleaning validation function - validates cleaned tables.
    
    This function is called by the Airflow DAG to validate cleaned data
    after processing to ensure quality standards are met.
    
    Returns:
        bool: True if validation passes, raises exception if it fails
    """
    anomaly_detector = AnomalyDetection()
    return anomaly_detector.run_post_validation()

def main(use_cleaned_tables=False):
    """
    Main function called by Airflow DAG for data validation.
    
    Args:
        use_cleaned_tables (bool): If True, validate cleaned tables; if False, validate source tables
        
    Returns:
        bool: True if validation passes, raises exception if it fails
    """
    anomaly_detector = AnomalyDetection()
    if use_cleaned_tables:
        return anomaly_detector.run_post_validation()
    else:
        return anomaly_detector.run_pre_validation()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()