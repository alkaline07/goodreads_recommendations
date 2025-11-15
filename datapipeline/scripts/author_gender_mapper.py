from google.cloud import bigquery
from gender_guesser.detector import Detector
from tqdm import tqdm
import sys
from datapipeline.scripts.logger_setup import get_logger
import time
from datetime import datetime
import os

class AuthorGenderMapper:
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
    
    def run(self):
        """
        Execute the author gender mapping.
        """
        # Initialize pipeline execution with logging
        self.logger.info("=" * 60)
        self.logger.info("Good Reads Author Gender Pipeline")
        start_time = time.time()
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        # Create author gender mapping for bias analysis
        self.create_author_gender_map()
        
        # Log completion
        end_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total runtime: {(end_time - start_time):.2f} seconds")
        self.logger.info("=" * 60)


def main():
    """
    Main entry point for the author gender mapping script.
    
    This function is called by the Airflow DAG to execute the gender mapping pipeline.
    It creates a AuthorGenderMapper instance and runs the complete mapping process.
    """
    gender_mapper = AuthorGenderMapper()
    gender_mapper.run()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()