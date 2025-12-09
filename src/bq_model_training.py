"""
BigQuery ML Model Training Module - FIXED VERSION
Handles concurrent model training and provides better error handling
"""

import os
import time
from datetime import datetime
from google.cloud import bigquery
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from datapipeline.scripts.logger_setup import get_logger
logger = get_logger("bq-model-training")
root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(root_env)
print("Loaded .env from:", root_env)


def safe_mlflow_log(func, *args, **kwargs):
    """Safely log to MLflow, continue if it fails."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning("MLflow logging warning", error=str(e))
        return None


class BigQueryMLModelTraining:
    """
    Fixed class for training BigQuery ML models with concurrency handling.
    """

    def __init__(self):
        """
        Initialize BigQuery ML model training.
        """
        if os.environ.get("AIRFLOW_HOME"):
            # Running locally or through Airflow
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get(
                "AIRFLOW_HOME") + "/gcp_credentials.json"

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # Table references
        self.train_table = f"{self.project_id}.{self.dataset_id}.goodreads_train_set"
        self.val_table = f"{self.project_id}.{self.dataset_id}.goodreads_validation_set"
        self.test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"

        # Model naming
        self.matrix_factorization_model = f"{self.project_id}.{self.dataset_id}.matrix_factorization_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.boosted_tree_model = f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.automl_regressor_model = f"{self.project_id}.{self.dataset_id}.automl_regressor_model"
        self.popularity_model = f"{self.project_id}.{self.dataset_id}.popularity_baseline"

    def check_and_cleanup_existing_models(self):
        """
        Check for existing models and running jobs, cleanup if needed.
        """
        try:
            logger.info("Checking for existing models and running jobs")

            # Check for running jobs
            jobs_query = """
            SELECT
                job_id,
                state,
                creation_time,
                statement_type
            FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
            WHERE statement_type = 'CREATE_MODEL'
                AND state IN ('PENDING', 'RUNNING')
            ORDER BY creation_time DESC
            LIMIT 10
            """

            try:
                running_jobs = self.client.query(jobs_query).to_dataframe(
                    create_bqstorage_client=False)
                if not running_jobs.empty:
                    logger.warning("Found running model training jobs, waiting",
                                   count=len(running_jobs))
                    time.sleep(30)  # Wait 30 seconds
            except Exception as e:
                logger.warning("Could not check running jobs", error=str(e))

            # List existing models
            models_to_check = [
                self.matrix_factorization_model.split('.')[-1],
                self.boosted_tree_model.split('.')[-1],
                self.automl_regressor_model.split('.')[-1]
            ]

            for model_name in models_to_check:
                try:
                    # Try to get model info
                    model_ref = f"{self.project_id}.{self.dataset_id}.{model_name}"
                    self.client.get_model(model_ref)
                    logger.info("Model exists", model=model_name)

                    # Optionally delete existing model
                    # self.client.delete_model(model_ref)
                    # print(f"Deleted existing model {model_name}")

                except Exception:
                    logger.info("Model does not exist", model=model_name)

        except Exception as e:
            logger.error("Error checking existing models", error=str(e))

    def safe_train_model(self, model_name, query, model_type, max_retries=3):
        """
        Safely train a model with retry logic and error handling.

        Args:
            model_name: Full model path
            query: CREATE MODEL query
            model_type: Type of model for logging
            max_retries: Maximum number of retry attempts
        """
        for attempt in range(max_retries):
            try:
                logger.info("Training model", model_type=model_type,
                            attempt=f"{attempt + 1}/{max_retries}")

                # Add a unique job ID to prevent conflicts
                job_config = bigquery.QueryJobConfig()
                job_id_prefix = f"{model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                job = self.client.query(query, job_config=job_config, job_id_prefix=job_id_prefix)

                # Wait with timeout
                result = job.result(timeout=3600)  # 1 hour timeout

                logger.info("Model training completed successfully", model_type=model_type)
                return True

            except Exception as e:
                error_msg = str(e)

                if "multiple create model query jobs" in error_msg.lower():
                    logger.warning("Model being updated by another job, waiting",
                                   attempt=attempt + 1)
                    time.sleep(60 * (attempt + 1))  # Exponential backoff

                elif "already exists" in error_msg.lower():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_model_name = f"{model_name}_{timestamp}"
                    query = query.replace(model_name, new_model_name)
                    logger.warning("Model already exists, using timestamp suffix",
                                   new_model_name=new_model_name)

                elif attempt < max_retries - 1:
                    logger.warning("Error training model, retrying", model_type=model_type,
                                   error=str(e), attempt=attempt + 1)
                    time.sleep(30 * (attempt + 1))

                else:
                    logger.error("Failed to train model after max retries", model_type=model_type,
                                 max_retries=max_retries, error=str(e))
                    return False

        return False

    def train_matrix_factorization(self):
        """
        Train MATRIX_FACTORIZATION model with error handling.
        """
        try:
            logger.info("Starting Matrix Factorization training")

            hyperparams = {
                "model_type": "MATRIX_FACTORIZATION",
                "l2_reg": 0.05,
                "num_factors": 25,
                "max_iterations": 40,
                "feedback_type": "EXPLICIT"
            }

            # Log hyperparameters to MLflow
            safe_mlflow_log(mlflow.log_params, {
                "mf_l2_reg": hyperparams["l2_reg"],
                "mf_num_factors": hyperparams["num_factors"],
                "mf_max_iterations": hyperparams["max_iterations"]
            })

            query = f"""
            CREATE OR REPLACE MODEL `{self.matrix_factorization_model}`
            OPTIONS(
                model_type='MATRIX_FACTORIZATION',
                model_registry='VERTEX_AI',
                vertex_ai_model_id='goodreads_matrix_factorization',
                user_col='user_id_clean',
                item_col='book_id',
                rating_col='rating',
                feedback_type='EXPLICIT',
                l2_reg=0.05,
                num_factors=25,
                max_iterations=40
            ) AS
            SELECT
                user_id_clean,
                book_id,
                rating
            FROM `{self.train_table}`
            WHERE rating IS NOT NULL
            """

            start_time = time.time()
            success = self.safe_train_model(
                self.matrix_factorization_model,
                query,
                "MATRIX_FACTORIZATION"
            )
            training_time = time.time() - start_time

            if success:
                safe_mlflow_log(mlflow.log_metric, "mf_training_time_seconds", training_time)
                self.evaluate_model(self.matrix_factorization_model, "MATRIX_FACTORIZATION")

            safe_mlflow_log(mlflow.log_param, "mf_model_name", self.matrix_factorization_model)
            safe_mlflow_log(mlflow.log_metric, "mf_training_success", 1 if success else 0)
            return success

        except Exception as e:
            logger.error("Unexpected error in train_matrix_factorization", error=str(e))
            safe_mlflow_log(mlflow.log_metric, "mf_training_success", 0)
            return False

    def train_boosted_tree_regressor(self):
        """
        Train BOOSTED_TREE_REGRESSOR model with error handling.
        """
        try:
            logger.info("Starting Boosted Tree Regressor training")

            feature_columns = self.get_feature_columns()
            feature_list = ", ".join(feature_columns)

            hyperparams = {
                "num_parallel_tree": 10,
                "max_tree_depth": 6,
                "subsample": 0.8,
                "min_split_loss": 0.001,
                "l1_reg": 0.01,
                "l2_reg": 0.05,
                "early_stop": True,
                "min_rel_progress": 0.005
            }

            # Log hyperparameters to MLflow
            safe_mlflow_log(mlflow.log_params, {
                "bt_num_parallel_tree": hyperparams["num_parallel_tree"],
                "bt_max_tree_depth": hyperparams["max_tree_depth"],
                "bt_subsample": hyperparams["subsample"],
                "bt_min_split_loss": hyperparams["min_split_loss"],
                "bt_l1_reg": hyperparams["l1_reg"],
                "bt_l2_reg": hyperparams["l2_reg"],
                "bt_num_features": len(feature_columns)
            })

            query = f"""
            CREATE OR REPLACE MODEL `{self.boosted_tree_model}`
            OPTIONS(
                model_type='BOOSTED_TREE_REGRESSOR',
                input_label_cols=['rating'],
                model_registry='VERTEX_AI',
                vertex_ai_model_id='goodreads_boosted_tree_regressor',
                num_parallel_tree=10,
                max_tree_depth=6,
                subsample=0.8,
                min_split_loss=0.001,
                l1_reg=0.01,
                l2_reg=0.05,
                early_stop=True,
                min_rel_progress=0.005
            ) AS
            SELECT
                {feature_list},
                rating
            FROM `{self.train_table}`
            WHERE rating IS NOT NULL
            """

            start_time = time.time()
            success = self.safe_train_model(
                self.boosted_tree_model,
                query,
                "BOOSTED_TREE_REGRESSOR"
            )
            training_time = time.time() - start_time

            if success:
                safe_mlflow_log(mlflow.log_metric, "bt_training_time_seconds", training_time)
                self.evaluate_model(self.boosted_tree_model, "BOOSTED_TREE_REGRESSOR")

            safe_mlflow_log(mlflow.log_param, "bt_model_name", self.boosted_tree_model)
            safe_mlflow_log(mlflow.log_metric, "bt_training_success", 1 if success else 0)
            return success

        except Exception as e:
            logger.error("Unexpected error in train_boosted_tree_regressor", error=str(e))
            safe_mlflow_log(mlflow.log_metric, "bt_training_success", 0)
            return False

    def get_feature_columns(self):
        """Get feature columns with error handling."""
        try:
            table = self.client.get_table(self.train_table)
            all_columns = [field.name for field in table.schema]

            exclude_columns = {
                'user_id_clean', 'book_id', 'rating', 'is_read',
                'user_days_to_read', 'user_book_recency'
            }

            feature_columns = [col for col in all_columns if col not in exclude_columns]
            logger.info("Feature columns retrieved", count=len(feature_columns))
            return feature_columns

        except Exception as e:
            logger.error("Error getting feature columns", error=str(e))
            # Return a default set
            return [
                'num_books_read', 'avg_rating_given', 'user_activity_count',
                'recent_activity_days', 'user_avg_reading_time_days',
                'title_clean', 'average_rating', 'adjusted_average_rating',
                'great', 'ratings_count', 'log_ratings_count',
                'popularity_score', 'book_popularity_normalized',
                'num_genres', 'is_series', 'title_length_in_characters',
                'title_length_in_words', 'description_length', 'num_pages',
                'publication_year', 'book_age_years', 'book_length_category',
                'book_era', 'avg_pages_per_day', 'avg_book_reading_time_days',
                'num_readers_with_reading_time', 'reading_pace_category',
                'user_avg_rating_vs_book', 'user_reading_speed_ratio',
                'user_pages_per_day_this_book', 'interaction_weight'
            ]

    def evaluate_model(self, model_name, model_type):
        """Evaluate model with error handling."""
        try:
            logger.info("Evaluating model", model_type=model_type)

            eval_query = f"""
            SELECT *
            FROM ML.EVALUATE(MODEL `{model_name}`, (
                SELECT *
                FROM `{self.val_table}`
                WHERE rating IS NOT NULL
                LIMIT 5000
            ))
            """

            eval_result = self.client.query(eval_query).to_dataframe(create_bqstorage_client=False)

            if not eval_result.empty:
                prefix = "mf_" if "MATRIX_FACTORIZATION" in model_type else "bt_"
                metrics = {}

                for col in eval_result.columns:
                    value = eval_result[col].iloc[0]
                    if isinstance(value, float):
                        metrics[col] = round(value, 4)
                        # Log metrics to MLflow
                        safe_mlflow_log(mlflow.log_metric, f"{prefix}{col.lower()}", value)

                logger.info("Model evaluation completed", model_type=model_type, metrics=metrics)

        except Exception as e:
            logger.error("Could not evaluate model", model_type=model_type, error=str(e))

    def analyze_data_characteristics(self):
        """
        Analyze the training data to understand its characteristics.
        This helps in setting appropriate hyperparameters.
        """
        try:
            logger.info("Analyzing training data characteristics")

            stats_query = f"""
            SELECT
                COUNT(DISTINCT user_id_clean) as num_users,
                COUNT(DISTINCT book_id) as num_books,
                COUNT(*) as num_interactions,
                AVG(rating) as avg_rating,
                STDDEV(rating) as std_rating,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating,
                APPROX_QUANTILES(rating, 4) as rating_quartiles
            FROM `{self.train_table}`
            WHERE rating IS NOT NULL
            """

            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)

            stats_dict = {col: stats[col].iloc[0] for col in stats.columns if
                          col != 'rating_quartiles'}
            logger.info("Training data statistics", stats=stats_dict)

            # Store for later use
            self.data_stats = stats.iloc[0].to_dict()

            # Check for cold-start problem
            cold_start_query = f"""
            WITH user_counts AS (
                SELECT user_id_clean, COUNT(*) as cnt
                FROM `{self.train_table}`
                GROUP BY user_id_clean
            )
            SELECT
                COUNTIF(cnt < 5) as users_with_few_ratings,
                COUNT(*) as total_users,
                COUNTIF(cnt < 5) / COUNT(*) as cold_start_ratio
            FROM user_counts
            """

            cold_start = self.client.query(cold_start_query).to_dataframe(
                create_bqstorage_client=False)
            cold_start_ratio = cold_start['cold_start_ratio'].iloc[0]
            logger.info("Cold-start analysis", ratio=f"{cold_start_ratio:.2%}",
                        users_with_few_ratings=cold_start['users_with_few_ratings'].iloc[0])

        except Exception as e:
            logger.error("Error analyzing data", error=str(e))
            self.data_stats = {}

    def run(self):
        """Execute the training pipeline with proper error handling."""
        start_time = time.time()
        logger.info("BigQuery ML Model Training Pipeline started",
                    timestamp=datetime.now().isoformat())

        # Start MLflow run
        experiment_name = "bigquery_ml_training"
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning("MLflow initialization warning, continuing anyway", error=str(e))

        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log run metadata
            safe_mlflow_log(mlflow.log_param, "project_id", self.project_id)
            safe_mlflow_log(mlflow.log_param, "dataset_id", self.dataset_id)
            safe_mlflow_log(mlflow.log_param, "train_table", self.train_table)
            safe_mlflow_log(mlflow.log_param, "val_table", self.val_table)
            safe_mlflow_log(mlflow.log_param, "test_table", self.test_table)

            # Check and cleanup if needed
            self.check_and_cleanup_existing_models()

            # Analyze data characteristics
            self.analyze_data_characteristics()

            # Log data statistics if available
            if hasattr(self, 'data_stats') and self.data_stats:
                for key, value in self.data_stats.items():
                    if isinstance(value, (int, float)) and key != 'rating_quartiles':
                        safe_mlflow_log(mlflow.log_metric, f"data_{key}", value)

            # Train models
            mf_success = self.train_matrix_factorization()
            bt_success = self.train_boosted_tree_regressor()

            # Summary
            end_time = time.time()
            total_runtime = end_time - start_time

            # Log final metrics
            safe_mlflow_log(mlflow.log_metric, "total_runtime_seconds", total_runtime)
            safe_mlflow_log(mlflow.log_metric, "all_models_success",
                            1 if (mf_success and bt_success) else 0)

            mlflow_run_id = None
            try:
                mlflow_run_id = mlflow.active_run().info.run_id
            except Exception:
                pass

            logger.info("Training completed",
                        timestamp=datetime.now().isoformat(),
                        mf_success=mf_success,
                        bt_success=bt_success,
                        total_runtime_seconds=round(total_runtime, 2),
                        mlflow_run_id=mlflow_run_id)


def main():
    """Main function with error handling."""
    try:
        trainer = BigQueryMLModelTraining()
        trainer.run()
    except Exception as e:
        logger.error("Fatal error in training pipeline", error=str(e))
        raise


if __name__ == "__main__":
    main()