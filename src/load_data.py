import pandas_gbq
import os
from google.cloud import bigquery
import pandas as pd
import bigframes.pandas as bpd

class DataLoader:
    def __init__(self):
        if not os.environ.get("AIRFLOW_HOME"):
            # Running in Github Actions
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"
        else:
            # Running locally
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

        # Initialize BigQuery client and get project information
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id="books"
        bpd.options.bigquery.project = self.project_id

    def load_data(self) -> bpd.DataFrame:
        """Load train data from BigQuery using the provided SQL query."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.goodreads_train_set`
        """
        # return pandas_gbq.read_gbq(query, project_id=self.project_id)
        return bpd.read_gbq(query)

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data()
    df.to_parquet("data/train_data.parquet", index=False)
    # df.to_csv("data/train_data.csv", index=False)
    # print(df.head())
    