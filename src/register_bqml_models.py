"""
This script finds the latest BigQuery ML models trained by the training script
and registers them in the Google Cloud Vertex AI Model Registry.

It handles versioning by finding existing models in the registry with a
matching display name and uploading the BQML model as a new version.
"""

import os
from google.cloud import aiplatform
from google.cloud import bigquery
from google.api_core import exceptions

class RegisterBQMLModels:
    """
    Class to handle registration of BQML models in Vertex AI Model Registry.
    """
    def __init__(self):
        # These are the "parent" model names in the Vertex AI Registry.
        # Each run will add a new *version* to these models.
        self.MODEL_REGISTRY_NAMES = {
            "BOOSTED_TREE": "goodreads_boosted_tree_regressor"
        }

        # These prefixes must match the ones in your training script
        # (e.g., matrix_factorization_model_... & boosted_tree_regressor_model_...)
        self.MODEL_PREFIXES = {
            "BOOSTED_TREE": "boosted_tree_regressor_model_"
        }


    def get_serving_image(self, model_type: str) -> str | None: # <-- Note the new return type
        """Gets the correct pre-built serving container for a BQML model."""
        if model_type == "BOOSTED_TREE":
            # BQML Boosted Tree is compatible with the BQML XGBoost container
            return "us-docker.pkg.dev/vertex-ai/prediction/bqml_xgboost:latest"
        else:
            raise ValueError(f"Unknown model type for serving image: {model_type}")


    def find_latest_model(self, bq_client: bigquery.Client, project_id: str, dataset_id: str, model_prefix: str) -> str | None:
        """
        Finds the most recently created BQML model with a given prefix
        using the client library's list_models() method.
        
        Args:
            bq_client: An initialized BigQuery client.
            project_id: The GCP project ID.
            dataset_id: The BigQuery dataset ID (e.g., 'books').
            model_prefix: The prefix of the model name (e.g., 'matrix_factorization_model_').

        Returns:
            The full model ID (project.dataset.model_name) or None if not found.
        """
        print(f"Searching for models with prefix '{model_prefix}' in {project_id}.{dataset_id}...")
        try:
            # Get a reference to the dataset.
            # The bq_client will automatically find it in the correct 'US' location.
            dataset_ref = bq_client.dataset(dataset_id, project=project_id)
            
            # Use the list_models() method instead of SQL
            models = list(bq_client.list_models(dataset_ref))
            
            if not models:
                print(f"No models found in dataset {project_id}.{dataset_id}")
                return None

            # Filter the models in Python to find the latest one with the prefix
            latest_model = None
            latest_time = 0  # Use 0 as the initial timestamp

            for model in models:
                if model.model_id.startswith(model_prefix):
                    model_timestamp = model.created.timestamp()
                    if model_timestamp > latest_time:
                        latest_time = model_timestamp
                        latest_model = model

            if latest_model:
                full_model_id = f"{project_id}.{dataset_id}.{latest_model.model_id}"
                print(f"Found latest model: {full_model_id} (created at {latest_model.created})")
                return full_model_id
            else:
                print(f"No model found with prefix '{model_prefix}' in {project_id}.{dataset_id}")
                return None
                
        except Exception as e:
            # This will catch permissions errors or if the dataset truly doesn't exist
            print(f"Error listing models with prefix '{model_prefix}': {e}")
            return None


    def register_model_in_vertex_ai(self, project_id: str, vertex_region: str, bqml_model_id: str, display_name: str, model_type: str):
        """
        Registers a BQML model in the Vertex AI Model Registry.
        ...
        """
        aiplatform.init(project=project_id, location=vertex_region)
        
        bq_model_uri = f"bq://{bqml_model_id}"
        serving_image = self.get_serving_image(model_type)
        
        print(f"Registering model '{display_name}' from {bq_model_uri}...")

        # ... (The 'parent_model' try/except block remains the same) ...
        parent_model = None
        try:
            existing_models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                location=vertex_region
            )
            if existing_models:
                parent_model = existing_models[0].resource_name
                print(f"Found existing model. Will register as new version of: {parent_model}")
        except exceptions.NotFound:
            print(f"No existing model found with display name '{display_name}'. Creating new model.")
        except Exception as e:
            print(f"Could not check for existing models: {e}")

        try:
            # We can go back to the simple version now
            model = aiplatform.Model.upload(
                display_name=display_name,
                serving_container_image_uri=serving_image,
                artifact_uri=f"bq://{bqml_model_id}",
                parent_model=parent_model,
                is_default_version=True,
                description=f"BQML {model_type} model registered from bq://{bqml_model_id}"
            )
            
            print(f"Successfully registered model: {model.resource_name}")
            print(f"View in Vertex AI Model Registry: {model.versioned_resource_name}")

        except Exception as e:
            print(f"Failed to register model {bq_model_uri}: {e}")
            print("Please ensure the Vertex AI API is enabled and that the BQML model exists.")

    def main(self):
        if os.environ.get("AIRFLOW_HOME"):
            # Running locally or through Airflow
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        # Initialize BigQuery client and get project information
        bq_client = bigquery.Client()
        project_id = bq_client.project
        dataset_id="books"
        
        for model_type, prefix in self.MODEL_PREFIXES.items():
            print(f"--- Processing {model_type} ---")
            
            # 1. Find the latest trained BQML model from your dataset
            latest_bqml_id = self.find_latest_model(
                bq_client=bq_client,
                project_id=project_id,
                dataset_id=dataset_id,
                model_prefix=prefix
            )
            
            if not latest_bqml_id:
                print(f"Could not find a model for {prefix}. Skipping registration.")
                continue
                
            # 2. Register this model in Vertex AI Model Registry
            try:
                registry_display_name = self.MODEL_REGISTRY_NAMES[model_type]
                self.register_model_in_vertex_ai(
                    project_id=project_id,
                    vertex_region="us-central1",
                    bqml_model_id=latest_bqml_id,
                    display_name=registry_display_name,
                    model_type=model_type
                )
            except Exception as e:
                print(f"Failed to register model {latest_bqml_id}: {e}")

        print("Model registration process complete.")


if __name__ == "__main__":
    registrar = RegisterBQMLModels()
    registrar.main()