import os
from google.cloud import bigquery
from datapipeline.scripts.logger_setup import get_logger

class ETLInteractions:
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
        self.source_table_id="user_interactions"
        self.destination_table_id="goodreads_interactions_mystery_thriller_crime"
        self.full_source_table_id = f"{self.project_id}.{self.dataset_id}.{self.source_table_id}"
        self.full_destination_table_id = f"{self.project_id}.{self.dataset_id}.{self.destination_table_id}"
    
    def backfill_rows(self):
        backfill_query = f"""
            UPDATE `{self.full_destination_table_id}`
            SET interaction_type = CASE
                -- Priority 1: If they finished the book
                WHEN is_read = TRUE THEN 'read'
                
                -- Priority 2: If they rated it but is_read is somehow false
                WHEN rating IS NOT NULL AND rating > 0 THEN 'like'
                
                -- Priority 3: Default to 'add_to_list' (added to list/library)
                ELSE 'add_to_list'
            END
            WHERE interaction_type IS NULL
        """
        try:
            query_job = self.client.query(backfill_query)
            query_job.result()
            self.logger.info("Backfill Complete. All old records now have an interaction_type.")
        
        except Exception as e:
            self.logger.error(f"Update Failed: {e}")
    

    def migrate_raw_events(self):
        self.logger.info(f"Moving raw event data to {self.full_destination_table_id}...")

        # 1. Ensure 'interaction_type' exists in destination
        table = self.client.get_table(self.full_destination_table_id)
        existing_cols = [f.name for f in table.schema]
        
        if "interaction_type" not in existing_cols:
            self.logger.info("Adding column: interaction_type")
            new_schema = table.schema[:] + [bigquery.SchemaField("interaction_type", "STRING", mode="NULLABLE")]
            table.schema = new_schema
            self.client.update_table(table, ["schema"])
            self.backfill_rows()  # Backfill existing rows after adding column

        # 2. The Migration Query
        # We populate the mandatory legacy fields with defaults, and put the real value in 'interaction_type'
        migration_query = f"""
            INSERT INTO `{self.full_destination_table_id}` (
                user_id, 
                book_id, 
                review_id, 
                date_added, 
                date_updated, 
                read_at, 
                started_at, 
                is_read,              
                rating,               
                review_text_incomplete,
                interaction_type     
            )
            SELECT
                user_id,
                SAFE_CAST(book_id AS INT64) as book_id,
                '' as review_id,
                
                CAST(event_timestamp AS STRING) as date_added,
                CAST(event_timestamp AS STRING) as date_updated,
                '' as read_at,
                '' as started_at,
                
                FALSE as is_read,
                0 as rating,          
                '' as review_text_incomplete,
                
                event_type as interaction_type
                
            FROM `{self.full_source_table_id}`
            WHERE event_type != 'view'
            
            -- Deduplication Logic using QUALIFY
            -- Partition by User+Book, Order by Priority (High to Low) then Time (New to Old)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY user_id, book_id 
                ORDER BY 
                    CASE 
                        WHEN event_type = 'like' THEN 3
                        WHEN event_type = 'add_to_list' THEN 2
                        WHEN event_type = 'click' THEN 1
                        ELSE 0 
                    END DESC,
                    event_timestamp DESC
            ) = 1
        """
        try:
            job = self.client.query(migration_query)
            job.result()
            self.logger.info(f"Migration Complete. Raw event types are stored in 'interaction_type'.")

            self.logger.info(f"Wiping source table {self.full_source_table_id}...")
            wipe_query = f"DELETE FROM `{self.full_source_table_id}` WHERE TRUE"
            wipe_job = self.client.query(wipe_query)
            wipe_job.result()
            self.logger.info("Source table wiped clean.")
            
        except Exception as e:
            self.logger.error(f"Error: {e}")

def main():
    etl = ETLInteractions()
    etl.migrate_raw_events()

if __name__ == "__main__":
    # Allow the script to be run directly for testing or development
    main()