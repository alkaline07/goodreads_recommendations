"""
BigQuery ML Model Training Module - FIXED VERSION
Handles concurrent model training and provides better error handling
"""

import os
import time
from datetime import datetime
from google.cloud import bigquery
import mlflow


def safe_mlflow_log(func, *args, **kwargs):
    """Safely log to MLflow, continue if it fails."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"MLflow logging warning: {e}")
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
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"

        # Table references
        self.train_table = f"{self.project_id}.{self.dataset_id}.goodreads_train_set"
        self.val_table = f"{self.project_id}.{self.dataset_id}.goodreads_validation_set"
        self.test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"

        # Model naming
        self.matrix_factorization_model = f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
        self.boosted_tree_model = f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"
        self.automl_regressor_model = f"{self.project_id}.{self.dataset_id}.automl_regressor_model"
        self.popularity_model = f"{self.project_id}.{self.dataset_id}.popularity_baseline"

    def check_and_cleanup_existing_models(self):
        """
        Check for existing models and running jobs, cleanup if needed.
        """
        try:
            print("Checking for existing models and running jobs...")
           
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
                running_jobs = self.client.query(jobs_query).to_dataframe(create_bqstorage_client=False)
                if not running_jobs.empty:
                    print(f"Found {len(running_jobs)} running model training jobs")
                    print("Waiting for existing jobs to complete...")
                    time.sleep(30)  # Wait 30 seconds
            except Exception as e:
                print(f"Could not check running jobs: {e}")
           
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
                    print(f"Model {model_name} exists")
                   
                    # Optionally delete existing model
                    # self.client.delete_model(model_ref)
                    # print(f"Deleted existing model {model_name}")
                   
                except Exception:
                    print(f"Model {model_name} does not exist")
                   
        except Exception as e:
            print(f"Error checking existing models: {e}")

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
                print(f"Training {model_type} model (attempt {attempt + 1}/{max_retries})...")
               
                # Add a unique job ID to prevent conflicts
                job_config = bigquery.QueryJobConfig()
                job_id_prefix = f"{model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
               
                job = self.client.query(query, job_config=job_config, job_id_prefix=job_id_prefix)
               
                # Wait with timeout
                result = job.result(timeout=3600)  # 1 hour timeout
               
                print(f"{model_type} model training completed successfully")
                return True
               
            except Exception as e:
                error_msg = str(e)
               
                if "multiple create model query jobs" in error_msg.lower():
                    print(f"Model is being updated by another job. Waiting...")
                    time.sleep(60 * (attempt + 1))  # Exponential backoff
                   
                elif "already exists" in error_msg.lower():
                    print(f"Model already exists. Using timestamp suffix...")
                    # Modify model name with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_model_name = f"{model_name}_{timestamp}"
                    query = query.replace(model_name, new_model_name)
                    print(f"Retrying with new model name: {new_model_name}")
                   
                elif attempt < max_retries - 1:
                    print(f"Error training {model_type}: {e}. Retrying...")
                    time.sleep(30 * (attempt + 1))
                   
                else:
                    print(f"Failed to train {model_type} after {max_retries} attempts: {e}")
                    return False
                   
        return False

    def train_matrix_factorization(self):
        """
        Train MATRIX_FACTORIZATION model with error handling.
        """
        try:
            print("=" * 60)
            print("Training MATRIX_FACTORIZATION Model")
            print("=" * 60)

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
            print(f"Unexpected error in train_matrix_factorization: {e}", exc_info=True)
            safe_mlflow_log(mlflow.log_metric, "mf_training_success", 0)
            return False

    def train_boosted_tree_regressor(self):
        """
        Train BOOSTED_TREE_REGRESSOR model with error handling.
        """
        try:
            print("=" * 60)
            print("Training BOOSTED_TREE_REGRESSOR Model")
            print("=" * 60)

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
            print(f"Unexpected error in train_boosted_tree_regressor: {e}", exc_info=True)
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
            print(f"Found {len(feature_columns)} feature columns")
            return feature_columns
           
        except Exception as e:
            print(f"Error getting feature columns: {e}")
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
                'user_pages_per_day_this_book'
            ]

    def evaluate_model(self, model_name, model_type):
        """Evaluate model with error handling."""
        try:
            print(f"Evaluating {model_type} model...")
           
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
                print(f"{model_type} Evaluation Metrics:")
                prefix = "mf_" if "MATRIX_FACTORIZATION" in model_type else "bt_"
                
                for col in eval_result.columns:
                    value = eval_result[col].iloc[0]
                    if isinstance(value, float):
                        print(f"  {col}: {value:.4f}")
                        # Log metrics to MLflow
                        safe_mlflow_log(mlflow.log_metric, f"{prefix}{col.lower()}", value)
                       
        except Exception as e:
            print(f"Could not evaluate {model_type}: {e}")
    
    def analyze_data_characteristics(self):
        """
        Analyze the training data to understand its characteristics.
        This helps in setting appropriate hyperparameters.
        """
        try:
            print("Analyzing training data characteristics...")
           
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
           
            print("Training Data Statistics:")
            for col in stats.columns:
                if col != 'rating_quartiles':
                    print(f"  {col}: {stats[col].iloc[0]}")
           
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
           
            cold_start = self.client.query(cold_start_query).to_dataframe(create_bqstorage_client=False)
            print(f"Cold-start ratio: {cold_start['cold_start_ratio'].iloc[0]:.2%} of users have < 5 ratings")
           
        except Exception as e:
            print(f"Error analyzing data: {e}")
            self.data_stats = {}


    def run(self):
        """Execute the training pipeline with proper error handling."""
        start_time = time.time()
        print("=" * 60)
        print("BigQuery ML Model Training Pipeline")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Start MLflow run
        experiment_name = "bigquery_ml_training"
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"MLflow initialization warning: {e}. Continuing with MLflow tracking (errors will be handled gracefully).")

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
            safe_mlflow_log(mlflow.log_metric, "all_models_success", 1 if (mf_success and bt_success) else 0)
            
            print("=" * 60)
            print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Matrix Factorization: {'SUCCESS' if mf_success else 'FAILED'}")
            print(f"Boosted Tree: {'SUCCESS' if bt_success else 'FAILED'}")
            print(f"Total runtime: {total_runtime:.2f} seconds")
            try:
                print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            except Exception:
                pass
            print("=" * 60)


def main():
    """Main function with error handling."""
    try:
        trainer = BigQueryMLModelTraining()
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()