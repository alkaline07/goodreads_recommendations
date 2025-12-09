"""
Minimal in-memory training stub used by local experiments and CI smoke-tests.

This module intentionally stays lightweight: it demonstrates how to plug a
`pandas.DataFrame` into downstream training logic without depending on BigQuery.
It is mostly leveraged by notebooks or quick validation scripts before running
the full BigQuery ML pipeline.
"""

import pandas as pd
from datapipeline.scripts.logger_setup import get_logger

logger = get_logger("df-model-training")

class DataFrameModelTraining:
    """Wrapper class that mimics the public API of the BigQuery trainer."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Fully materialized training data used for experimentation.
        """
        self.df = df

    def train_model(self) -> None:
        """
        Placeholder training hook; replace with scikit-learn / XGBoost logic as needed.

        The stub logs shape information to make CI logs and notebooks explicit.
        """
        logger.info(f"Training model with shape {self.df.shape}")


if __name__ == "__main__":
    # For demonstration purposes, load the parquet artifact produced by `load_data.py`.
    demo_df = pd.read_parquet("data/train_data.parquet")
    trainer = DataFrameModelTraining(demo_df)
    trainer.train_model()