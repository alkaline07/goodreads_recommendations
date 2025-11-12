"""
Book Recommendation Script for All Users
Gets top 5 recommendations from both Matrix Factorization and Boosted Tree models
"""

import os
from google.cloud import bigquery
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookRecommender:
    """Generate book recommendations for all users using ML models."""
   
    def __init__(self, credentials_path=None):
        """Initialize BigQuery client and model references."""
       
        # Set credentials
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
       
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
       
        # Table references
        self.features_table = f"{self.project_id}.{self.dataset_id}.goodreads_features"
        self.train_table = f"{self.project_id}.{self.dataset_id}.goodreads_train_set"
        self.books_table = f"{self.project_id}.{self.dataset_id}.goodreads_books_cleaned"
       
        # Model references - update these to match your actual model names
        self.mf_model = self.find_latest_model('matrix_factorization_model')
        self.bt_model = self.find_latest_model('boosted_tree_regressor_model')
       
        logger.info(f"Initialized recommender for project: {self.project_id}")
   
    def find_latest_model(self, model_prefix: str) -> str:

        """Find the latest version of a model by prefix."""

        try:

            query = f"""

            SELECT 

                model_name,

                creation_time

            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.MODELS`

            WHERE model_name LIKE '{model_prefix}%'

            ORDER BY creation_time DESC

            LIMIT 1

            """

            result = self.client.query(query).to_dataframe()

            if not result.empty:

                model_name = result['model_name'].iloc[0]

                full_path = f"{self.project_id}.{self.dataset_id}.{model_name}"

                logger.info(f"Found model: {full_path}")

                return full_path

            else:

                # Fallback to base name

                logger.warning(f"No model found with prefix {model_prefix}, trying base name")

                return f"{self.project_id}.{self.dataset_id}.{model_prefix}"

        except Exception as e:

            logger.error(f"Error finding model: {e}")

            # Fallback to base name

            return f"{self.project_id}.{self.dataset_id}.{model_prefix}"
 
    def get_all_users(self, limit=None) -> List[str]:
        """Get all unique users from the dataset."""
       
        query = f"""
        SELECT DISTINCT user_id_clean
        FROM `{self.features_table}`
        WHERE user_id_clean IS NOT NULL
        """
       
        if limit:
            query += f" LIMIT {limit}"
       
        logger.info("Fetching all unique users...")
        users_df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
        users = users_df['user_id_clean'].tolist()
        logger.info(f"Found {len(users)} unique users")
       
        return users
   
    def check_user_in_training(self, user_id: str) -> bool:
        """Check if user exists in training data (needed for Matrix Factorization)."""
       
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.train_table}`
        WHERE user_id_clean = '{user_id}'
        """
       
        result = self.client.query(query).to_dataframe(create_bqstorage_client=False)
        return result['count'].iloc[0] > 0
   
    def get_mf_recommendations(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        """Get recommendations using Matrix Factorization model."""
       
        try:
            # Check if user is in training data
            if not self.check_user_in_training(user_id):
                logger.warning(f"User {user_id} not in training data - MF cannot make predictions")
                return pd.DataFrame()
           
            query = f"""
            WITH user_read_books AS (
                SELECT DISTINCT book_id
                FROM `{self.features_table}`
                WHERE user_id_clean = '{user_id}'
            ),
            predictions AS (
                SELECT
                    book_id,
                    predicted_rating
                FROM ML.PREDICT(MODEL `{self.mf_model}`,
                    (
                        SELECT
                            '{user_id}' as user_id_clean,
                            book_id
                        FROM (
                            SELECT DISTINCT book_id
                            FROM `{self.features_table}`
                            WHERE book_id NOT IN (SELECT book_id FROM user_read_books)
                            AND book_id IS NOT NULL
                            LIMIT 100  -- Limit candidates for efficiency
                        )
                    )
                )
            )
            SELECT
                p.book_id,
                f.title_clean,
                p.predicted_rating as mf_score,
                f.ratings_count,
                'matrix_factorization' as model_type
            FROM predictions p
            LEFT JOIN (
                SELECT DISTINCT book_id,
                       ANY_VALUE(title_clean) as title_clean,
                       ANY_VALUE(ratings_count) as ratings_count
                FROM `{self.features_table}`
                GROUP BY book_id
            ) f ON p.book_id = f.book_id
            ORDER BY p.predicted_rating DESC
            LIMIT {top_k}
            """
           
            result = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return result
           
        except Exception as e:
            logger.error(f"Error getting MF recommendations for user {user_id}: {e}")
            return pd.DataFrame()
   
    def get_bt_recommendations(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        """Get recommendations using Boosted Tree model."""
       
        try:
            query = f"""
            WITH user_features AS (
                SELECT
                    MAX(num_books_read) as num_books_read,
                    MAX(avg_rating_given) as avg_rating_given,
                    MAX(user_activity_count) as user_activity_count,
                    MAX(recent_activity_days) as recent_activity_days,
                    MAX(user_avg_reading_time_days) as user_avg_reading_time_days
                FROM `{self.features_table}`
                WHERE user_id_clean = '{user_id}'
            ),
            user_read_books AS (
                SELECT DISTINCT book_id
                FROM `{self.features_table}`
                WHERE user_id_clean = '{user_id}'
            ),
            predictions AS (
                SELECT
                    book_id,
                    title_clean,
                    predicted_rating,
                    ratings_count
                FROM ML.PREDICT(
                    MODEL `{self.bt_model}`,
                    (
                        SELECT
                            'synthetic' as user_id_clean,
                            f.book_id,
                            0.0 as rating,
                            false as is_read,
                            0 as user_days_to_read,
                            0 as user_book_recency,
                            u.num_books_read,
                            u.avg_rating_given,
                            u.user_activity_count,
                            u.recent_activity_days,
                            u.user_avg_reading_time_days,
                            f.title_clean,
                            f.average_rating,
                            f.adjusted_average_rating,
                            f.great,
                            f.ratings_count,
                            f.log_ratings_count,
                            f.popularity_score,
                            f.book_popularity_normalized,
                            f.num_genres,
                            f.is_series,
                            f.title_length_in_characters,
                            f.title_length_in_words,
                            f.description_length,
                            f.num_pages,
                            f.publication_year,
                            f.book_age_years,
                            f.book_length_category,
                            f.book_era,
                            f.avg_pages_per_day,
                            f.avg_book_reading_time_days,
                            f.num_readers_with_reading_time,
                            f.reading_pace_category,
                            f.user_avg_rating_vs_book,
                            f.user_reading_speed_ratio,
                            f.user_pages_per_day_this_book
                        FROM `{self.features_table}` f
                        CROSS JOIN user_features u
                        WHERE f.book_id NOT IN (SELECT book_id FROM user_read_books)
                        AND f.ratings_count > 50
                        LIMIT 100  -- Limit candidates for efficiency
                    )
                )
            )
            SELECT
                book_id,
                title_clean,
                predicted_rating as bt_score,
                ratings_count,
                PERCENT_RANK() OVER (ORDER BY predicted_rating) * 100 as percentile,
                'boosted_tree' as model_type
            FROM predictions
            ORDER BY predicted_rating DESC
            LIMIT {top_k}
            """
           
            result = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return result
           
        except Exception as e:
            logger.error(f"Error getting BT recommendations for user {user_id}: {e}")
            return pd.DataFrame()
   
    def get_combined_recommendations(self, user_id: str, top_k: int = 5) -> Dict:
        """Get recommendations from both models for a user."""
       
        logger.info(f"Getting recommendations for user: {user_id}")
       
        # Get recommendations from both models
        mf_recs = self.get_mf_recommendations(user_id, top_k)
        bt_recs = self.get_bt_recommendations(user_id, top_k)
       
        # Combine results
        result = {
            'user_id': user_id,
            'matrix_factorization': mf_recs.to_dict('records') if not mf_recs.empty else [],
            'boosted_tree': bt_recs.to_dict('records') if not bt_recs.empty else [],
            'timestamp': datetime.now().isoformat()
        }
       
        return result
   
    def generate_recommendations_batch(self, users: List[str] = None,
                                      sample_size: int = None,
                                      output_file: str = None) -> pd.DataFrame:
        """
        Generate recommendations for multiple users.
       
        Args:
            users: List of user IDs (if None, gets all users)
            sample_size: If set, randomly samples this many users
            output_file: If set, saves results to CSV file
       
        Returns:
            DataFrame with all recommendations
        """
       
        # Get users if not provided
        if users is None:
            users = self.get_all_users(limit=sample_size)
        elif sample_size and len(users) > sample_size:
            import random
            users = random.sample(users, sample_size)
       
        logger.info(f"Generating recommendations for {len(users)} users...")
       
        all_recommendations = []
       
        for i, user_id in enumerate(users, 1):
            if i % 10 == 0:
                logger.info(f"Processing user {i}/{len(users)}...")
           
            try:
                # Get recommendations
                recs = self.get_combined_recommendations(user_id)
               
                # Format for DataFrame
                for rec in recs.get('matrix_factorization', []):
                    all_recommendations.append({
                        'user_id': user_id,
                        'model': 'matrix_factorization',
                        'rank': len([r for r in recs['matrix_factorization'] if r['mf_score'] > rec['mf_score']]) + 1,
                        'book_id': rec['book_id'],
                        'title': rec.get('title_clean', 'Unknown'),
                        'score': rec['mf_score'],
                        'ratings_count': rec.get('ratings_count', 0)
                    })
               
                for rec in recs.get('boosted_tree', []):
                    all_recommendations.append({
                        'user_id': user_id,
                        'model': 'boosted_tree',
                        'rank': len([r for r in recs['boosted_tree'] if r['bt_score'] > rec['bt_score']]) + 1,
                        'book_id': rec['book_id'],
                        'title': rec.get('title_clean', 'Unknown'),
                        'score': rec['bt_score'],
                        'percentile': rec.get('percentile', 0),
                        'ratings_count': rec.get('ratings_count', 0)
                    })
               
                # Add small delay to avoid overwhelming BigQuery
                if i % 50 == 0:
                    time.sleep(1)
                   
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                continue
       
        # Create DataFrame
        df_results = pd.DataFrame(all_recommendations)
       
        # Save to file if requested
        if output_file:
            df_results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
       
        # Print summary statistics
        self.print_summary(df_results)
       
        return df_results
   
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics of recommendations."""
       
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATION SUMMARY")
        logger.info("="*60)
       
        # Overall stats
        n_users = df['user_id'].nunique()
        n_books = df['book_id'].nunique()
       
        logger.info(f"Total users processed: {n_users}")
        logger.info(f"Unique books recommended: {n_books}")
       
        # Model-specific stats
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            logger.info(f"\n{model.upper()} Model:")
            logger.info(f"  - Users with recommendations: {model_df['user_id'].nunique()}")
            logger.info(f"  - Total recommendations: {len(model_df)}")
            logger.info(f"  - Unique books: {model_df['book_id'].nunique()}")
           
            if 'score' in model_df.columns:
                logger.info(f"  - Score range: [{model_df['score'].min():.3f}, {model_df['score'].max():.3f}]")
                logger.info(f"  - Mean score: {model_df['score'].mean():.3f}")
       
        # Most recommended books
        logger.info("\nTop 10 Most Recommended Books:")
        top_books = df.groupby(['book_id', 'title']).size().reset_index(name='count')
        top_books = top_books.sort_values('count', ascending=False).head(10)
       
        for _, row in top_books.iterrows():
            logger.info(f"  - {row['title'][:50]}: {row['count']} times")
       
        logger.info("="*60)
   
    def export_to_bigquery(self, df: pd.DataFrame, table_name: str):
        """Export recommendations to BigQuery table."""
       
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
       
        # Configure the load job
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",  # Replace table
            autodetect=True
        )
       
        # Load DataFrame to BigQuery
        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for job to complete
       
        logger.info(f"Exported {len(df)} recommendations to {table_id}")


def main():
    """Main function to generate recommendations."""
   
    # Initialize recommender
    recommender = BookRecommender(
        credentials_path="config/gcp_credentials.json"  # Update path as needed
    )
   
    # Option 1: Get recommendations for a sample of users
    logger.info("Generating recommendations for sample users...")
    df_results = recommender.generate_recommendations_batch(
        sample_size=100,  # Start with 10 users for testing
        output_file="data/recommendations_output.csv"
    )
   
    # Option 2: Get recommendations for specific users
    # specific_users = ['0d09d5132aba9100eb117b808705a0ac', 'another_user_id']
    # df_results = recommender.generate_recommendations_batch(
    #     users=specific_users,
    #     output_file="specific_users_recommendations.csv"
    # )
   
    # Option 3: Export to BigQuery for analysis
    # recommender.export_to_bigquery(df_results, "user_recommendations")
   
    # Display sample results
    if not df_results.empty:
        logger.info("\nSample recommendations:")
        print(df_results.head(20))
   
    logger.info("\nRecommendation generation complete!")


if __name__ == "__main__":
    main()