"""
Generate Prediction Tables for Bias Detection

This script:
1. Finds the latest trained models
2. Generates predictions with all features needed for bias detection
3. Creates BigQuery tables optimized for bias analysis
4. Includes slicing features (popularity, era, length, etc.)

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from google.cloud import bigquery
from datetime import datetime
from typing import List, Dict, Optional
import time


class BiasReadyPredictionGenerator:
    """
    Generate prediction tables with all features needed for bias detection.
    """
    
    def __init__(self):
        """Initialize the generator."""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"
        
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        
        print(f"BiasReadyPredictionGenerator initialized")
        print(f"Project: {self.project_id}")
        print(f"Dataset: {self.dataset_id}\n")
    
    def find_latest_model(self, model_prefix: str) -> Optional[str]:
        """
        Find the latest version of a model.
        
        Args:
            model_prefix: Model name prefix (e.g., 'boosted_tree_regressor_model')
            
        Returns:
            Full model path or None if not found
        """
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            models = list(self.client.list_models(dataset_ref))
            
            matching_models = [
                model for model in models 
                if model.model_id.startswith(model_prefix)
            ]
            
            if matching_models:
                matching_models.sort(key=lambda m: m.created, reverse=True)
                latest_model = matching_models[0]
                full_path = f"{self.project_id}.{self.dataset_id}.{latest_model.model_id}"
                print(f"✓ Found model: {latest_model.model_id}")
                print(f"  Created: {latest_model.created}")
                return full_path
            else:
                print(f"✗ No model found with prefix: {model_prefix}")
                return None
                
        except Exception as e:
            print(f"✗ Error finding model: {e}")
            return None
    
    def verify_test_table_exists(self) -> bool:
        """Verify that the test set table exists."""
        test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"
        
        try:
            table = self.client.get_table(test_table)
            
            # Check row count
            count_query = f"SELECT COUNT(*) as cnt FROM `{test_table}` WHERE rating IS NOT NULL"
            count = self.client.query(count_query).to_dataframe(create_bqstorage_client=False)['cnt'].iloc[0]
            
            print(f"✓ Test table exists: {test_table}")
            print(f"  Rows with ratings: {count:,}\n")
            return True
            
        except Exception as e:
            print(f"✗ Test table not found: {test_table}")
            print(f"  Error: {e}\n")
            return False
    
    def generate_boosted_tree_predictions(self, model_path: str) -> bool:
        """
        Generate predictions from boosted tree model with bias detection features.
        
        Args:
            model_path: Full path to the model
            
        Returns:
            True if successful
        """
        print("\n" + "="*80)
        print("GENERATING BOOSTED TREE PREDICTIONS FOR BIAS DETECTION")
        print("="*80 + "\n")
        
        output_table = f"{self.project_id}.{self.dataset_id}.boosted_tree_rating_predictions"
        test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"
        
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        SELECT
          pred.user_id_clean,
          pred.book_id,
          pred.rating AS actual_rating,
          pred.predicted_rating,
          ABS(pred.rating - pred.predicted_rating) AS absolute_error,
          (pred.rating - pred.predicted_rating) AS error,
          -- Slicing features for bias detection
          pred.book_popularity_normalized,
          pred.book_length_category,
          pred.book_era,
          pred.num_genres,
          pred.user_activity_count,
          pred.reading_pace_category,
          pred.average_rating,
          pred.ratings_count,
          pred.num_pages,
          pred.publication_year
        FROM ML.PREDICT(
          MODEL `{model_path}`,
          (
            SELECT
              num_books_read, avg_rating_given, user_activity_count, 
              recent_activity_days, user_avg_reading_time_days, 
              title_clean, average_rating, adjusted_average_rating, 
              great, ratings_count, log_ratings_count, 
              popularity_score, book_popularity_normalized, 
              num_genres, is_series, 
              title_length_in_characters, title_length_in_words, 
              description_length, num_pages, 
              publication_year, book_age_years, 
              book_length_category, book_era, 
              avg_pages_per_day, avg_book_reading_time_days, 
              num_readers_with_reading_time, reading_pace_category, 
              user_avg_rating_vs_book, user_reading_speed_ratio, 
              user_pages_per_day_this_book,
              rating,
              user_id_clean,
              book_id
            FROM `{test_table}`
            WHERE rating IS NOT NULL
          )
        ) AS pred
        """
        
        try:
            print(f"Model: {model_path}")
            print(f"Output: {output_table}")
            print("\nExecuting query (this may take 2-5 minutes)...")
            
            job = self.client.query(query)
            job.result()
            
            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as num_predictions,
                AVG(absolute_error) as mean_absolute_error,
                SQRT(AVG(POWER(error, 2))) as root_mean_squared_error,
                AVG(predicted_rating) as avg_predicted_rating,
                AVG(actual_rating) as avg_actual_rating
            FROM `{output_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            print("\n✓ Boosted Tree Predictions Generated Successfully!")
            print("\nStatistics:")
            print(f"  Predictions: {stats['num_predictions'].iloc[0]:,.0f}")
            print(f"  MAE: {stats['mean_absolute_error'].iloc[0]:.4f}")
            print(f"  RMSE: {stats['root_mean_squared_error'].iloc[0]:.4f}")
            print(f"  Avg Predicted: {stats['avg_predicted_rating'].iloc[0]:.4f}")
            print(f"  Avg Actual: {stats['avg_actual_rating'].iloc[0]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error generating predictions: {e}")
            return False
    
    def generate_automl_predictions(self, model_path: str) -> bool:
        """
        Generate predictions from AutoML model.
        
        Args:
            model_path: Full path to the model
            
        Returns:
            True if successful
        """
        print("\n" + "="*80)
        print("GENERATING AUTOML PREDICTIONS FOR BIAS DETECTION")
        print("="*80 + "\n")
        
        output_table = f"{self.project_id}.{self.dataset_id}.automl_rating_predictions"
        test_table = f"{self.project_id}.{self.dataset_id}.goodreads_test_set"
        
        query = f"""
        CREATE OR REPLACE TABLE `{output_table}` AS
        SELECT
          pred.user_id_clean,
          pred.book_id,
          pred.rating AS actual_rating,
          pred.predicted_rating,
          ABS(pred.rating - pred.predicted_rating) AS absolute_error,
          (pred.rating - pred.predicted_rating) AS error,
          -- Slicing features
          pred.book_popularity_normalized,
          pred.book_length_category,
          pred.book_era,
          pred.num_genres,
          pred.user_activity_count,
          pred.reading_pace_category
        FROM ML.PREDICT(
          MODEL `{model_path}`,
          (
            SELECT *
            FROM `{test_table}`
            WHERE rating IS NOT NULL
          )
        ) AS pred
        """
        
        try:
            print(f"Model: {model_path}")
            print(f"Output: {output_table}")
            print("\nExecuting query (this may take 2-5 minutes)...")
            
            job = self.client.query(query)
            job.result()
            
            # Get statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as num_predictions,
                AVG(absolute_error) as mean_absolute_error,
                SQRT(AVG(POWER(error, 2))) as root_mean_squared_error
            FROM `{output_table}`
            """
            stats = self.client.query(stats_query).to_dataframe(create_bqstorage_client=False)
            
            print("\n✓ AutoML Predictions Generated Successfully!")
            print("\nStatistics:")
            print(f"  Predictions: {stats['num_predictions'].iloc[0]:,.0f}")
            print(f"  MAE: {stats['mean_absolute_error'].iloc[0]:.4f}")
            print(f"  RMSE: {stats['root_mean_squared_error'].iloc[0]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error generating predictions: {e}")
            return False
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the dataset."""
        print("\n" + "="*80)
        print("SCANNING FOR AVAILABLE MODELS")
        print("="*80 + "\n")
        
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            models = list(self.client.list_models(dataset_ref))
            
            model_types = {
                'boosted_tree': [],
                'matrix_factorization': [],
                'automl': []
            }
            
            for model in models:
                model_id = model.model_id
                if 'boosted_tree' in model_id.lower():
                    model_types['boosted_tree'].append(model_id)
                elif 'matrix_factorization' in model_id.lower():
                    model_types['matrix_factorization'].append(model_id)
                elif 'automl' in model_id.lower():
                    model_types['automl'].append(model_id)
            
            print("Available Models:")
            for model_type, model_list in model_types.items():
                if model_list:
                    print(f"\n  {model_type.upper()}:")
                    for model_id in sorted(model_list):
                        print(f"    • {model_id}")
            
            if not any(model_types.values()):
                print("  No models found!")
            
            print()
            return model_types
            
        except Exception as e:
            print(f"✗ Error listing models: {e}\n")
            return {}
    
    def run(self, model_types: List[str] = None):
        """
        Run the complete prediction generation pipeline.
        
        Args:
            model_types: List of model types to generate predictions for
                        ['boosted_tree', 'automl'] or None for all
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print("BIAS-READY PREDICTION GENERATION PIPELINE")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Step 1: List available models
        available_models = self.list_available_models()
        
        # Step 2: Verify test table exists
        if not self.verify_test_table_exists():
            print("✗ Cannot proceed without test table")
            return
        
        # Step 3: Determine which models to process
        if model_types is None:
            model_types = ['boosted_tree', 'automl']
        
        results = {}
        
        # Step 4: Generate predictions for each model type
        if 'boosted_tree' in model_types and available_models.get('boosted_tree'):
            model_path = self.find_latest_model('boosted_tree_regressor_model')
            if model_path:
                results['boosted_tree'] = self.generate_boosted_tree_predictions(model_path)
            else:
                results['boosted_tree'] = False
        
        if 'automl' in model_types and available_models.get('automl'):
            model_path = self.find_latest_model('automl_regressor_model')
            if model_path:
                results['automl'] = self.generate_automl_predictions(model_path)
            else:
                results['automl'] = False
        
        # Summary
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "="*80)
        print("PREDICTION GENERATION COMPLETE")
        print("="*80)
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
        print("\n--- Results ---")
        for model_type, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {model_type.upper()}: {status}")
        
        print("\n--- Next Steps ---")
        if any(results.values()):
            print("  1. Run bias detection:")
            print("     cd src && python bias_detection.py")
            print("\n  2. Or run full bias audit pipeline:")
            print("     cd src && python bias_pipeline.py")
        else:
            print("  ✗ No predictions were generated successfully")
            print("  Check errors above and ensure models are trained")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function."""
    generator = BiasReadyPredictionGenerator()
    
    # Generate predictions for all available models
    generator.run(model_types=['boosted_tree', 'automl'])


if __name__ == "__main__":
    main()
