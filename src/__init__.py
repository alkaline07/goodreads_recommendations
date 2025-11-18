"""
Core Goodreads recommendation system package.

This namespace exposes the end-to-end ML pipeline utilities used across:
• Data ingestion (`load_data`)
• Model training (`bq_model_training`, `df_model_training`)
• Bias analysis (`bias_detection`, `bias_mitigation`, `bias_pipeline`)
• Model governance (`model_evaluation_pipeline`, `model_manager`, `model_selector`)
• Deployment support (`generate_prediction_tables`, `register_bqml_models`)

Importing this package makes the shared types and helper modules discoverable for
Airflow DAGs, notebooks, and CLIs that orchestrate the project workflows.
"""

__all__ = [
    "bias_detection",
    "bias_mitigation",
    "bias_pipeline",
    "bias_visualization",
    "bq_model_training",
    "df_model_training",
    "generate_prediction_tables",
    "load_data",
    "model_evaluation_pipeline",
    "model_manager",
    "model_selector",
    "model_sensitivity_analysis",
    "model_validation",
    "register_bqml_models",
]
