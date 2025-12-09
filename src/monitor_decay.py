import sys
from google.cloud import bigquery
import os

class MonitorDecay:
    """
    This script checks for model performance decay by calculating the Click-Through Rate (CTR)
    over the last 24 hours. If the CTR falls below a defined threshold, it triggers an alert.
    """
    def __init__(self):
        if os.environ.get("AIRFLOW_HOME"):
            # Running locally or through Airflow
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        self.table_id = "user_interactions"
        self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        # Threshold: If CTR is below 1.5%, trigger alert
        self.CTR_THRESHOLD = 0.20

    def check_model_decay(self):
        print("--- Starting Model Decay Check ---")
        
        # Query: Calculate CTR for the last 24 hours
        query = f"""
            SELECT
                AVG(SAFE_DIVIDE(user_clicks, user_views)) as avg_ctr,
                COUNT(*) as distinct_users_counted
            FROM (
                SELECT
                    user_id,
                    COUNTIF(event_type = 'view') as user_views,
                    COUNTIF(event_type = 'click') as user_clicks
                FROM `{self.full_table_id}`
                WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
                GROUP BY user_id
            )
            WHERE user_clicks > 0
        """

        try:
            job = self.client.query(query)
            result = list(job.result()) # Wait for job to complete
            
            if not result or not result[0].avg_ctr:
                print("No data found for the last 24 hours.")
                sys.exit(0)
                return

            row = result[0]
            avg_ctr = row.avg_ctr
            user_count = row.distinct_users_counted

            print(f"Metrics (Last 24h): Avg Per-User CTR={avg_ctr:.4f} (calculated from {user_count} users)")

            if avg_ctr < self.CTR_THRESHOLD:
                print(f"DECAY DETECTED! Avg CTR ({avg_ctr:.2%}) is below threshold ({self.CTR_THRESHOLD:.2%})")
                sys.exit(1) # Exit 1 to notify GitHub Actions that decay happened
            else:
                print(f"Model performance is healthy. Avg CTR ({avg_ctr:.2%}) is above threshold.")
                sys.exit(0)

        except Exception as e:
            print(f"Error running monitor: {e}")
            sys.exit(1)

if __name__ == "__main__":
    monitor = MonitorDecay()
    monitor.check_model_decay()