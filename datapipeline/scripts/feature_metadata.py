"""
Feature Metadata Collection Module for Goodreads Data Pipeline

This module collects and stores metadata about the generated features table
for data versioning and tracking purposes. It extracts schema information,
row counts, and other metadata to support MLOps practices.

Key Features:
- Collects table schema information from BigQuery
- Records row counts and data statistics
- Generates metadata files for data versioning
- Supports DVC (Data Version Control) integration
- Provides data lineage tracking

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import json
from datetime import datetime
from datapipeline.scripts.logger_setup import get_logger
from google.cloud import bigquery

class FeatureMetadata:

    def __init__(self):
        """
        Initialize the FeatureMetadata class with BigQuery client and configuration.
        
        Sets up:
        - Google Cloud credentials for BigQuery access
        - Logging configuration for metadata collection operations
        - BigQuery client and project information
        - Dataset reference for metadata collection
        """
        # Set Google Application Credentials for BigQuery access
        # Uses AIRFLOW_HOME environment variable to locate credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME", ".") + "/gcp_credentials.json"

        # Initialize logging for metadata collection operations
        self.logger = get_logger("feature_metadata")

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

    def run(self):
        """
        Collect and store metadata about the features table for data versioning.
        
        This method extracts comprehensive metadata including:
        - Table schema information (column names, types, modes)
        - Row count and data statistics
        - Timestamp and query hash for tracking
        - Saves metadata to JSON file for DVC integration
        """
        try:
            self.logger.info("Starting feature metadata collection")
            
            # Query to get row count from the features table
            query = f"SELECT COUNT(*) FROM `{self.project_id}.{self.dataset_id}.goodreads_features`"
            job = self.client.query(query)

            # Get table schema information from BigQuery
            table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.goodreads_features")
            schema = [
                {"name": field.name, "type": field.field_type, "mode": field.mode}
                for field in table.schema
            ]

            # Record current timestamp for metadata
            timestamp = datetime.now()

            # Get row count from query result
            row_count = job.to_dataframe(create_bqstorage_client=False).iloc[0, 0]

            # Create comprehensive metadata dictionary
            metadata = {
                "table": f"{self.dataset_id}.goodreads_features",
                "schema": schema,
                "row_count": int(row_count) if row_count is not None else None,
                "query_hash": hash(query),  # Hash of the query for tracking changes
                "timestamp": timestamp.isoformat()
            }

            # Create metadata directory and save metadata file
            os.makedirs("data/metadata", exist_ok=True)
            metadata_path = "data/metadata/goodreads_features_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info("Wrote metadata to file", path=metadata_path)
        except Exception:
            self.logger.exception("Failed to run feature metadata collection")
            raise

def main():
    """
    Main entry point for the feature metadata collection script.
    
    This function is called by the Airflow DAG to execute the metadata collection process.
    It creates a FeatureMetadata instance and runs the metadata collection pipeline.
    """
    feature_metadata = FeatureMetadata()
    feature_metadata.run()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()