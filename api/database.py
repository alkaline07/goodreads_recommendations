from google.cloud import bigquery
import os

def get_bq_client():
    airflow_home = os.environ.get("AIRFLOW_HOME")
    if airflow_home:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = airflow_home + "/gcp_credentials.json"
    return bigquery.Client()
