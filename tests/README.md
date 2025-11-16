# Test Suite for Goodreads Recommendations

This directory contains test cases for all Python modules in the `src/` folder.

## Test Files

- `test_bq_model_training.py` - Tests for BigQuery ML model training
- `test_bias_detection.py` - Tests for bias detection functionality
- `test_bias_mitigation.py` - Tests for bias mitigation techniques
- `test_bias_pipeline.py` - Tests for the integrated bias audit pipeline
- `test_bias_visualization.py` - Tests for bias visualization generation
- `test_df_model_training.py` - Tests for DataFrame-based model training
- `test_generate_bias_prediction_tables.py` - Tests for prediction table generation
- `test_load_data.py` - Tests for data loading functionality
- `test_model_evaluation_pipeline.py` - Tests for model evaluation pipeline
- `test_model_selector.py` - Tests for model selection based on performance and fairness
- `test_model_sensitivity_analysis.py` - Tests for feature importance analysis
- `test_model_validation.py` - Tests for model validation
- `test_register_bqml_models.py` - Tests for BigQuery ML model registration

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run a specific test file
```bash
pytest tests/test_bq_model_training.py
```

### Run a specific test class
```bash
pytest tests/test_bq_model_training.py::TestBigQueryMLModelTraining
```

### Run a specific test method
```bash
pytest tests/test_bq_model_training.py::TestBigQueryMLModelTraining::test_init
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

All tests use:
- **pytest** as the testing framework
- **unittest.mock** for mocking external dependencies (BigQuery, MLflow, etc.)
- **Fixtures** defined in `conftest.py` for common test setup

## Notes

- Tests mock external services (BigQuery, MLflow, Vertex AI) to avoid requiring actual credentials
- Tests focus on unit testing individual functions and methods
- Integration tests would require actual GCP credentials and resources

## Dependencies

Test dependencies are included in `requirements.txt`:
- pytest
- pytest-mock (if needed for advanced mocking)
- All production dependencies for import testing

