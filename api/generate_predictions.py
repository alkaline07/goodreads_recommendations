import os
from google.cloud import bigquery, aiplatform
from google.api_core.exceptions import NotFound
from typing import Optional, Dict
import datetime
import argparse
import json
from datapipeline.scripts.logger_setup import get_logger

logger = get_logger("generate-predictions")

class GeneratePredictions:
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
        self.location = "us-central1"
        aiplatform.init(
            project=self.project_id,
            location=self.location
        )

    def get_mf_predictions(self, model_name, user_id):
        """
        Generate book recommendations for a given user_id using Matrix Factorization model.
        """
        logger.info(f"Generating MF predictions for {user_id} on model {model_name}")
        config = {
            "project_id": self.project_id,
            "dataset": self.dataset_id,
            "model_name": model_name
        }
        
        with open("api/mf_predictor_query.sql", "r") as file:
            query_template = file.read()
        
        query = query_template.format(**config)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
        return results[['book_id', 'title', 'rating', 'author_names']]

    def get_bt_predictions(self, model_name, user_id):
        """
        Generate book recommendations for a given user_id using Boosted Tree model.
        """
        logger.info(f"Generating MF predictions for {user_id} on model {model_name}")
        config = {
            "project_id": self.project_id,
            "dataset": self.dataset_id,
            "model_name": model_name
        }
        
        with open("api/bt_predictor_query.sql", "r") as file:
            query_template = file.read()
        
        query = query_template.format(**config)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
        return results[['book_id', 'title', 'rating', 'author_names']]

    def get_selected_model_info(self) -> Dict[str, str]:
        """
        Get the selected model information from the model selection report.
        
        Returns:
            Dictionary with model_name and display_name
        """
        model_selection_path = "docs/bias_reports/model_selection_report.json"
        logger.info(f"Retrieving selected model info from {model_selection_path}")
        
        default_info = {
            "model_name": "boosted_tree_regressor",
            "display_name": "goodreads_boosted_tree_regressor"
        }
        
        try:
            with open(model_selection_path, 'r') as f:
                report = json.load(f)
            
            selected = report.get('selected_model', {})
            model_name = selected.get('model_name', 'boosted_tree_regressor')

            logger.info(f"Selected model found {model_name}")
            
            return {
                "model_name": model_name,
                "display_name": f"goodreads_{model_name}"
            }
        except FileNotFoundError:
            logger.warning("Model selection report not found, using default")
            return default_info
        except Exception as e:
            logger.error("Error reading model selection report", error=str(e))
            return default_info
    
    def get_version(self, display_name):
        logger.info(f"Retrieving model version for {display_name}")
        try:
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=self.location
            )
            
            if not models:
                logger.warning("No model found with display name", display_name=display_name)
                return None
            
            parent_model = models[0]
            logger.info("Found parent model", resource_name=parent_model.resource_name)
            
            versions = parent_model.versioning_registry.list_versions()
            default_version = None
                
            for v in versions:
                if hasattr(v, 'version_aliases') and 'default' in v.version_aliases:
                    default_version = v
                    break
            logger.info("Default version found", version_id=default_version.version_id if default_version else "None")
            return default_version.version_id if default_version else None
        except Exception as e:
            logger.error("Error retrieving model version", error=str(e))
            return None

    def get_bq_model_id_by_version(self, display_name, version_id):
        logger.info(f"Retrieving BigQuery model ID for {display_name} version {version_id}")
        
        dataset_ref = self.client.dataset(self.dataset_id)

        # 1. Determine Model Name Prefix based on display name
        if "boosted_tree" in display_name:
            prefix = "boosted_tree_regressor_model_"
        elif "matrix_factorization" in display_name:
            prefix = "matrix_factorization_model_"
        else:
            raise ValueError(f"Model type for {display_name} not recognized.")

        # 2. Try to construct the specific ID using the Vertex timestamp
        model_resource_name = f"projects/{self.project_id}/locations/{self.location}/models/{display_name}@{version_id}"
        model_version = aiplatform.Model(model_resource_name)
        model_dict = model_version.to_dict()

        # Get version creation time
        version_create_time = model_dict['versionCreateTime']
        # Convert to datetime and format as YYYYMMDD_HHMMSS
        create_datetime = datetime.datetime.fromisoformat(version_create_time.replace('Z', '+00:00'))
        timestamp_str = create_datetime.strftime("%Y%m%d_%H%M%S")
        
        specific_model_id = f"{prefix}{timestamp_str}"
        
        # 3. Check if this specific model exists
        try:
            model_ref = dataset_ref.model(specific_model_id)
            self.client.get_model(model_ref) # This triggers the API call
            logger.info(f"Success: Found exact match {specific_model_id}")
            return specific_model_id
            
        except NotFound:
            logger.info(f"Model {specific_model_id} not found (likely due to timestamp drift).")
            logger.info("Falling back to the latest available model...")

            # 4. Fallback: List all models and find the latest one matching the prefix
            models = list(self.client.list_models(dataset_ref))
            
            # Filter for models that belong to this group
            relevant_models = [m for m in models if m.model_id.startswith(prefix)]
            
            if not relevant_models:
                raise ValueError(f"No models found in dataset {self.dataset_id} starting with {prefix}")

            # Sort by creation time (descending) to get the newest
            relevant_models.sort(key=lambda x: x.model_id, reverse=True)
            
            latest_model_id = relevant_models[0].model_id
            logger.info(f"Returning latest model: {latest_model_id}")
            
            return f"{self.project_id}.{self.dataset_id}.{latest_model_id}"

    def get_model_from_registry(self, display_name: str) -> Optional[str]:
        logger.info(f"Getting model from registry for {display_name}")
        version_id = self.get_version(display_name)
        logger.info("Retrieved version ID {version_id}")
        if not version_id:
            logger.error("No version found for model", display_name=display_name)
            return None
        bq_model_id = self.get_bq_model_id_by_version(display_name, version_id)
        if not bq_model_id:
            logger.info("Falling back to default model registry path")
            if "matrix_factorization" in display_name:
                bq_model_id = f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
            elif "boosted_tree" in display_name:
                bq_model_id = f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"
            else:
                logger.error("Model type for display name not recognized", display_name=display_name)
                return None
        logger.info(f"Using BigQuery model ID {bq_model_id}")
        return bq_model_id
    
    def get_predictions(self, user_id):
        """
        Generate book recommendations for a given user_id using the selected model.
        """
        logger.info(f"Generating predictions for {user_id}")
        model_info = self.get_selected_model_info()
        if not model_info:
            raise ValueError("No model selected for predictions.")
        
        model_name = model_info['display_name']
        logger.info(f"Using model {model_name}")
        # model_name = "goodreads_matrix_factorization" # Uncomment to force MF model for testing

        bq_model_id = self.get_model_from_registry(model_name)

        if not bq_model_id:
            raise ValueError(f"Could not retrieve BigQuery model ID for model {model_name}.")
        if "matrix_factorization" in model_name:
            predictions = self.get_mf_predictions(bq_model_id, user_id)
        elif "boosted_tree" in model_name:
            predictions = self.get_bt_predictions(bq_model_id, user_id)
        else:
            raise ValueError(f"Model type for {model_name} not recognized.")     
        return predictions[:10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate book predictions for a user")
    parser.add_argument("--user_id", type=str, required=True, help="User ID to generate predictions for")
    
    args = parser.parse_args()
    
    generator = GeneratePredictions()
    predictions = generator.get_predictions(args.user_id)
    logger.info("Predictions generated", count=len(predictions))
    print(predictions)

# Sample runner command:
# python -m api.generate_predictions --user_id "017fa7fa5ca764f1b912b4b1716adca5"

