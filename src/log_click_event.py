import uuid
from datetime import datetime
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, NotFound
import os
import argparse
import time

class LogClickEvent:
    def __init__(self):
        if os.environ.get("AIRFLOW_HOME"):
            # Running locally or through Airflow
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        self.table_id = "user_interactions"
        self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    def _create_table_if_not_exists(self):
        """Internal function to ensure the table exists before writing."""
        try:
            self.client.get_table(self.full_table_id)
        except NotFound:
            print(f"Table {self.full_table_id} not found. Creating it...")
            schema = [
                bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("book_id", "INT64", mode="REQUIRED"),
                bigquery.SchemaField("book_title", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
            ]
            table = bigquery.Table(self.full_table_id, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="event_timestamp"
            )
            self.client.create_table(table)
            print("Table created successfully.")
    
    def _wait_for_table(self):
        for i in range(10):
            try:
                self.client.get_table(self.full_table_id)
                return True
            except NotFound:
                time.sleep(1)
        return False

    def log_user_event(self, user_id: str, book_id: int, event_type: str, book_title: str = None):
        """
        Call this function from your API backend to log data to BigQuery.
        """
        # 1. Ensure table exists (You might want to move this to app startup to save time per request)
        self._create_table_if_not_exists()
        if not self._wait_for_table():
            raise RuntimeError("Finding User Interaction table timed out.")

        if event_type not in ["read", "click", "view", "like", "add_to_list", "similar"]:
            raise ValueError("Invalid event_type. Must be one of read, click, view, like, add_to_list, similar")

        if event_type == "similar":
            return True  # Do not log 'similar' events type, it does not represent user action

        # 2. Prepare data
        rows_to_insert = [{
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "book_id": book_id,
            "book_title": book_title,
            "event_type": event_type,
            "event_timestamp": datetime.utcnow().isoformat()
        }]

        # 3. Insert to BigQuery
        try:
            errors = self.client.insert_rows_json(self.full_table_id, rows_to_insert)
            if errors == []:
                print(f"Logged {event_type} for user {user_id}")
                return True
            else:
                print(f"Errors inserting rows: {errors}")
                return False
        except GoogleAPIError as e:
            print(f"API Error: {e}")
            return False
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log click events to BigQuery")
    parser.add_argument("--user_id", type=str, required=True, help="User ID")
    parser.add_argument("--book_id", type=int, required=True, help="Book ID")
    parser.add_argument("--event_type", type=str, required=True, help="Event type")
    parser.add_argument("--book_title", type=str, default=None, help="Book title (optional)")
    
    args = parser.parse_args()
    
    logger = LogClickEvent()
    logger.log_user_event(
        user_id=args.user_id,
        book_id=args.book_id,
        event_type=args.event_type,
        book_title=args.book_title
    )