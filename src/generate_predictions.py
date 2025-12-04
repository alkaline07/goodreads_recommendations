from src.model_deployment import get_selected_model_info, ModelDeployer
import os
from google.cloud import bigquery, aiplatform
import pandas as pd
from typing import Optional

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
        aiplatform.init(
            project=self.project_id,
            location="us-central1"
        )

    def get_mf_predictions(self, endpoint_name, user_id):
        """
        Generate book recommendations for a given user_id using Matrix Factorization model.
        """
        endpoint = aiplatform.Endpoint(endpoint_name)
        instances = {"user_id": user_id}
        print(instances)
        results = endpoint.predict(instances=instances).predictions[0]
        results_df = pd.DataFrame({
            'book_id': results['predicted_book_id'],
            'predicted_rating': results['predicted_rating']
        })
        results = results_df.sort_values(by='predicted_rating', ascending=False)
        return results

    def get_bt_predictions(self, model_name, user_id):
        """
        Generate book recommendations for a given user_id using Boosted Tree model.
        """
        config = {
            "project_id": self.project_id,
            "dataset": self.dataset_id,
            "model_name": model_name
        }
        
        with open("src/bt_predictor_query.sql", "r") as file:
            query_template = file.read()
        
        query = query_template.format(**config)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
        print(results.columns)
        return results[['book_id', 'title', 'rating', 'author_names']]
    
    def get_model_from_registry(self, display_name: str) -> Optional[str]:
        if "boosted_tree" in display_name:
            return f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"
        elif "matrix_factorization" in display_name:
            return "{self.project_id}.{self.dataset_id}.matrix_factorization_model"
        else:
            print(f"Model type for {display_name} not recognized.")
            return None

    
    def get_predictions(self, user_id):
        """
        Generate book recommendations for a given user_id using the selected model.
        """
        model_info = get_selected_model_info()
        if not model_info:
            raise ValueError("No model selected for predictions.")
        
        model_name = model_info['display_name']

        bq_model_id = self.get_model_from_registry(model_name)

        if not bq_model_id:
            raise ValueError(f"Could not retrieve BigQuery model ID for model {model_name}.")        
        predictions = self.get_bt_predictions(bq_model_id, user_id)
        
        return predictions

if __name__ == "__main__":
    generator = GeneratePredictions()
    user_id = "017fa7fa5ca764f1b912b4b1716adca5"  # Replace with actual user_id
    predictions = generator.get_predictions(user_id)
    print(predictions)


