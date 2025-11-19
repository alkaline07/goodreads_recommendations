"""
Lightweight data access helpers used by notebooks and local experiments.

The module focuses on a single responsibility: providing a typed utility that
knows how to connect to BigQuery, respect Airflow credential conventions, and
fetch the canonical Goodreads training dataset as a BigFrames DataFrame.
"""

import os

import bigframes.pandas as bpd
import pandas as pd
import pandas_gbq
from google.cloud import bigquery


class DataLoader:
    """Encapsulates credential bootstrapping and dataset retrieval logic."""

    def __init__(self):
        """
        Configure credentials and client state once so downstream consumers
        (scripts, notebooks, unit tests) do not rewrite this boilerplate.
        """
        airflow_home = os.environ.get("AIRFLOW_HOME")
        if airflow_home:
            # Allow Airflow workers and local runs to share the same credential file.
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{airflow_home}/gcp_credentials.json"

        # Initialize BigQuery client and record project metadata for query templating.
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # bigframes uses its own global config; align it with the active project.
        bpd.options.bigquery.project = self.project_id

    def load_data(self) -> bpd.DataFrame:
        """
        Materialize the curated Goodreads training set as a BigFrames DataFrame.

        Returns:
            bpd.DataFrame: Distributed BigFrames representation of the train split.
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.goodreads_train_set`
        """
        return bpd.read_gbq(query)

    # def load_to_pandas(self) -> pd.DataFrame:
    #     """
    #     Convenience helper for callers that prefer an in-memory pandas DataFrame.

    #     Returns:
    #         pandas.DataFrame: Fully materialized copy of the train split.
    #     """
    #     query = f"""
    #     SELECT *
    #     FROM `{self.project_id}.{self.dataset_id}.goodreads_train_set`
    #     """
    #     return pandas_gbq.read_gbq(query, project_id=self.project_id)


if __name__ == "__main__":
    loader = DataLoader()
    dataframe = loader.load_data()
    dataframe.to_parquet("data/train_data.parquet", index=False)