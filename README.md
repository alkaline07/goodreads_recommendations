# Book Recommendation System (Goodreads + MLOps)

This project builds a machine learning-based book recommendation system using Goodreads data, with an end-to-end MLOps-ready architecture. It includes data processing, model training, recommendation logic, an API for serving, and Docker for containerization.

## Team Members

- Ananya Asthana
- Arpita Wagulde  
- Karan Goyal  
- Purva Agarwal  
- Shivam Sah  
- Shivani Sharma

## Project Architecture Overview

![Project Architecture](assets/architecture.jpg)

## Data Sources

- **Goodreads Dataset:** Books, ratings, and metadata from the [Goodbooks dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)

Citations:
- Mengting Wan, Julian McAuley, ["Item Recommendation on Monotonic Behavior Chains"](https://mengtingwan.github.io/paper/recsys18_mwan.pdf), in RecSys'18.
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, ["Fine-Grained Spoiler Detection from Large-Scale Review Corpora"](https://mengtingwan.github.io/paper/acl19_mwan.pdf), in ACL'19.

## Getting Started

> **Note:** To run our project on Windows machine, ensure WSL is installed.

### 1. Clone the Repository

```bash
git clone https://github.com/purva-agarwal/goodreads_recommendations.git
cd goodreads_recommendations
```

### 2. Set up Python Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# OR
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

> **Note:** Make sure to activate your virtual environment before running any subsequent commands.

### 3. Run Data Pre-Processing Pipeline without Docker

- Set the following environment variables from terminal.
  The variable set is only for this instance of terminal and will not affect others.
  
  ```bash
  export AIRFLOW_HOME=/path/to/your/config/folder
  export AIRFLOW__SMTP__SMTP_MAIL_FROM="husky.mlops@gmail.com"
  export AIRFLOW__SMTP__SMTP_USER="husky.mlops@gmail.com"
  export AIRFLOW__SMTP__SMTP_PASSWORD=<SHARED_PASSWORD>
  ```
  
  Replace `/path/to/your/config/folder` with the absolute path to the config folder of the cloned repository.

- Request access to GCP credentials (access credentials will be shared per user basis)

- Place the access credentials in the config folder as `gcp_credentials.json`

```bash
airflow standalone
```

**Access the Airflow UI:**

A login password for the admin user will be shown in the terminal or in `config/simple_auth_manager_passwords.json.generated`

Open your browser and go to: <http://localhost:8080>

Login using the admin credentials

**Add GCP Connection on Airflow UI:**

1. Go to Admin → Connections
2. Click "Add Connection"
3. Set Connection ID: `goodreads_conn`
4. Set Connection Type: `Google Cloud`
5. Paste the shared GCP access credentials JSON in the "Extra Fields JSON" field



**Add Email Connection on Airflow UI:**
1. Admin >> Connections
2. Add Connection
3. Connection ID : smtp_default ,  Connection Type : Email
4. Add the following values in standard field,
    - Host : smtp.gmail.com
    - Login: husky.mlops@gmail.com
    - Port: 587
    - Password : <SHARED_PASSWORD>
5. Add the following JSON in the Extra fields JSON
```json
{
    "from_email": "husky.mlops@gmail.com"
}
```

**Run the DAG:**

1. In the Airflow UI, search for `goodreads_recommendation_pipeline` DAG
2. Click "Trigger DAG" to start execution

#### Locally (for development)

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to test the API using Swagger UI.


### 4. Run Data Pre-Processing Pipeline With Docker
  - Ensure that correct credentials are set in .env file
  - Run the docker compose up command : 
    ```bash
    docker compose up --build
    ```
  - Open your browser and go to: <http://localhost:8080> 
  - Login using the credentials 
    - username : admin
    - password : admin
  - Run the docker command to shutdown the containers 
    ```bash
    docker compose down
    ```

## GitHub Actions CI/CD Pipeline

Our project uses GitHub Actions to automate the complete ML pipeline from data loading to model deployment. The workflows are orchestrated in a sequential pipeline that ensures data quality, model training, evaluation, bias detection, validation, and version management.

### Pipeline Overview

The CI/CD pipeline consists of **7 sequential workflows** plus **2 supporting workflows**:

```text
1. Load Data
   ↓
2. Model Training
   ↓
3. Generate Predictions
   ↓
4. Evaluate Model
   ↓
5. Bias Detection & Mitigation Pipeline
   ↓
6. Model Validation
   ↓
7. Model Manager
```

**Supporting Workflows:**
- **PR Test Suite** - Runs on pull requests to ensure code quality
- **Send Email Notification** - Reusable workflow for sending notifications

### Workflow Details

#### 1. Load Data (`1_load_data.yml`)

**Purpose:** Loads training data from BigQuery and prepares it for model training.

**Trigger:** Manual workflow dispatch (can be triggered manually from GitHub Actions UI)

**Steps:**

1. **Checkout code** - Retrieves the repository code
2. **Set up Python 3.11** - Configures Python environment
3. **Authenticate to Google Cloud** - Uses GCP credentials from GitHub secrets to access BigQuery
4. **Install dependencies** - Installs packages from `model_requirements.txt`
5. **Load the data** - Executes `src.load_data` module to:
   - Connect to BigQuery
   - Extract training data from curated tables
   - Process and prepare data for training
   - Save data locally as `data/train_data.parquet`
6. **Upload data artifact** - Uploads the parquet file as a GitHub Actions artifact named `goodreads-train-data` (retained for 1 day)
7. **Notifications** - Sends success/failure notifications via email

**Output Artifacts:**
- `goodreads-train-data` artifact containing `data/train_data.parquet`

**Next Workflow:** Automatically triggers "2. Model Training" on success

---

#### 2. Model Training (`2_train_model.yml`)

**Purpose:** Trains machine learning models using BigQuery ML with MLflow tracking.

**Trigger:** Automatically runs after "1. Load Data" workflow completes successfully

**Steps:**

1. **Checkout code** - Retrieves repository code
2. **Set up Python 3.11** - Configures Python environment
3. **Install dependencies** - Installs required packages
4. **Authenticate to Google Cloud** - Sets up GCP authentication for BigQuery ML
5. **Download data artifact** - Downloads the `goodreads-train-data` artifact from the previous workflow using GitHub CLI
6. **Move data to correct location** - Places the parquet file in the `data/` directory
7. **Train the model BigQuery ML (with MLflow)** - This step:
   - Starts MLflow UI server in the background on port 5000
   - Waits for MLflow server to be ready (up to 30 seconds)
   - Executes `src.bq_model_training` which:
     - Trains models using BigQuery ML
     - Logs metrics, parameters, and artifacts to MLflow
     - Saves model artifacts
   - Automatically cleans up MLflow server on exit
8. **Commit and push new files** - Commits generated artifacts to `artifacts_bot` branch:
   - Model artifacts
   - MLflow tracking data
   - Any generated outputs
9. **Notifications** - Sends success/failure email notifications

**Key Features:**
- **MLflow Integration**: Automatic MLflow server management with health checks
- **Artifact Persistence**: All model artifacts are committed to a dedicated branch
- **BigQuery ML**: Leverages Google Cloud's managed ML service for scalable training

**Output Artifacts:**
- Trained models in BigQuery
- MLflow tracking data
- Model artifacts committed to `artifacts_bot` branch

**Next Workflow:** Automatically triggers "3. Generate Predictions" on success

---

#### 3. Generate Predictions (`3_generate_prediction_table.yml`)

**Purpose:** Generates prediction tables with bias-ready features for all trained models.

**Trigger:** 
- Automatically runs after "2. Model Training" completes successfully
- Can also be manually triggered via workflow dispatch

**Steps:**

1. **Checkout repository** - Retrieves code
2. **Set up Python 3.11** - Configures environment
3. **Authenticate to Google Cloud** - Sets up GCP access
4. **Install required Python dependencies** - Installs packages
5. **Run Prediction Generator** - Executes `src.generate_prediction_tables` which:
   - Loads trained models from BigQuery
   - Generates predictions for test/validation sets
   - Creates bias-ready prediction tables with:
     - Predicted ratings
     - Actual ratings
     - Feature values
     - Slice dimensions (popularity, era, gender, etc.)
   - Stores predictions in BigQuery tables
6. **Commit and push new files** - Commits prediction artifacts to `artifacts_bot` branch
7. **Notifications** - Sends email notifications

**Key Features:**
- **Bias-Ready Tables**: Predictions include all necessary dimensions for bias analysis
- **Multiple Models**: Generates predictions for all trained models
- **BigQuery Storage**: Predictions stored in BigQuery for efficient analysis

**Output Artifacts:**
- Prediction tables in BigQuery (e.g., `boosted_tree_rating_predictions`, `matrix_factorization_rating_predictions`)
- Artifacts committed to `artifacts_bot` branch

**Next Workflow:** Automatically triggers "4. Evaluate Model" on success

---

#### 4. Evaluate Model (`4_evaluate_model.yml`)

**Purpose:** Evaluates model performance and generates evaluation reports with feature importance analysis.

**Trigger:** Automatically runs after "3. Generate Predictions" completes successfully

**Steps:**

1. **Checkout repository** - Retrieves code
2. **Set up Python 3.11** - Configures environment
3. **Authenticate to Google Cloud** - Sets up GCP access
4. **Install dependencies** - Installs required packages
5. **Run Model Evaluation** - This step:
   - Starts MLflow UI server in the background
   - Waits for MLflow server readiness
   - Executes `src.model_evaluation_pipeline` which:
     - Computes performance metrics (MAE, RMSE, R², accuracy within thresholds)
     - Runs SHAP-based feature importance analysis
     - Generates evaluation reports in JSON format
     - Logs metrics to MLflow
     - Creates sensitivity analysis artifacts
   - Cleans up MLflow server
6. **Commit and push new files** - Commits evaluation reports and artifacts to `artifacts_bot` branch
7. **Notifications** - Sends email notifications

**Key Features:**
- **Comprehensive Metrics**: Multiple performance metrics for thorough evaluation
- **Feature Importance**: SHAP analysis for model interpretability
- **Report Generation**: JSON reports stored in `docs/model_analysis/evaluation/`

**Output Artifacts:**
- Evaluation reports (JSON) in `docs/model_analysis/evaluation/`
- Feature importance analysis in `docs/model_analysis/sensitivity/`
- MLflow logged metrics
- Artifacts committed to `artifacts_bot` branch

**Next Workflow:** Automatically triggers "5. Bias Detection & Mitigation Pipeline" on success

---

#### 5. Bias Detection & Mitigation Pipeline (`5_run_bias_pipeline.yml`)

**Purpose:** Detects bias across multiple dimensions and applies mitigation techniques if needed.

**Trigger:** Automatically runs after "4. Evaluate Model" completes successfully

**Steps:**

1. **Checkout repository** - Retrieves code
2. **Set up Python 3.11** - Configures environment
3. **Authenticate to Google Cloud** - Sets up GCP access
4. **Install dependencies** - Installs required packages
5. **Run Bias Detection & Mitigation Pipeline** - Executes `src.bias_pipeline` which:
   - **Bias Detection**: Analyzes predictions across 8 dimensions:
     - Book Popularity, Book Length, Book Era
     - Genre Diversity, User Activity, Reading Pace
     - Author Gender, Rating Range
   - **Disparity Analysis**: Computes MAE, RMSE, and fairness metrics per slice
   - **Mitigation Application**: Applies prediction shrinkage or other techniques if bias detected
   - **Validation**: Re-evaluates models post-mitigation
   - **Visualization Generation**: Creates fairness scorecards, heatmaps, and comparison charts
   - **Model Selection**: Compares models and selects best based on performance + fairness
   - **Report Generation**: Creates comprehensive audit reports
6. **Commit and push new files** - Commits all bias reports and visualizations to `artifacts_bot` branch:
   - Detection reports
   - Comprehensive audit reports
   - Mitigation reports
   - Visualizations (PNG files)
   - Model selection reports
   - Pipeline logs
7. **Notifications** - Sends email notifications

**Key Features:**
- **Comprehensive Bias Analysis**: 8 dimensions analyzed simultaneously
- **Automated Mitigation**: Applies mitigation techniques when bias is detected
- **Rich Visualizations**: Generates multiple visualization types for stakeholder review
- **Model Selection**: Balances accuracy with fairness

**Output Artifacts:**
- Bias detection reports in `docs/bias_reports/`
- Comprehensive audit reports
- Visualizations in `docs/bias_reports/visualizations/`
- Model selection report and comparison chart
- Pipeline logs in `docs/bias_reports/bias_pipeline_logs/`
- All artifacts committed to `artifacts_bot` branch

**Next Workflow:** Automatically triggers "6. Model Validation" on success

---

#### 6. Model Validation (`6_model_validation.yml`)

**Purpose:** Validates trained models against performance thresholds and ensures they meet deployment criteria.

**Trigger:** Automatically runs after "5. Bias Detection & Mitigation Pipeline" completes successfully

**Steps:**

1. **Checkout repository** - Retrieves code
2. **Set up Python 3.11** - Configures environment
3. **Authenticate to Google Cloud** - Sets up GCP access
4. **Install dependencies** - Installs required packages
5. **Validate model** - This step:
   - Starts MLflow UI server in the background
   - Waits for MLflow server readiness
   - Executes `src.model_validation` which:
     - Runs ML.EVALUATE on train/val/test splits
     - Checks RMSE thresholds
     - Validates performance metrics meet requirements
     - Persists validation reports in JSON format
     - Enforces quality gates before deployment
   - Cleans up MLflow server
6. **Commit and push new files** - Commits validation reports to `artifacts_bot` branch
7. **Notifications** - Sends success/failure notifications

**Key Features:**
- **Quality Gates**: Enforces minimum performance thresholds
- **Multi-Split Validation**: Validates on train, validation, and test sets
- **Automated Blocking**: Prevents deployment if validation fails

**Output Artifacts:**
- Validation reports (JSON)
- MLflow logged validation metrics
- Artifacts committed to `artifacts_bot` branch

**Validation Criteria:**
- RMSE must be below threshold
- MAE must meet requirements
- Performance must be consistent across splits

**Next Workflow:** Automatically triggers "7. Model Manager" on success

---

#### 7. Model Manager (`7_model_manager.yml`)

**Purpose:** Manages model versioning, compares new models with current production, and promotes/rolls back based on performance deltas.

**Trigger:** Automatically runs after "6. Model Validation" completes successfully

**Steps:**

1. **Checkout repository** - Retrieves code
2. **Set up Python 3.11** - Configures environment
3. **Authenticate to Google Cloud** - Sets up GCP access
4. **Install dependencies** - Installs required packages
5. **Manage the default model version** - This step:
   - Starts MLflow UI server in the background
   - Waits for MLflow server readiness
   - Executes `src.model_manager` which:
     - Compares new model version with current default
     - Calculates RMSE deltas
     - Decides whether to promote or roll back
     - Updates default model version in Vertex AI Model Registry
     - Manages model versioning and metadata
   - Cleans up MLflow server
6. **Commit and push new files** - Commits version management artifacts to `artifacts_bot` branch
7. **Notifications** - Sends success/failure notifications

**Key Features:**
- **Automated Promotion**: Promotes models that improve performance
- **Rollback Protection**: Prevents regression by comparing with current production
- **Version Control**: Integrates with Vertex AI Model Registry
- **Delta-Based Decisions**: Uses RMSE deltas to make promotion decisions

**Output Artifacts:**
- Model version management records
- Promotion/rollback decisions
- Artifacts committed to `artifacts_bot` branch

**Promotion Logic:**
- New model must have RMSE improvement (or within acceptable threshold)
- Fairness scores must meet minimum requirements
- Validation must have passed in previous step

---

### Supporting Workflows

#### PR Test Suite (`pr_tests.yml`)

**Purpose:** Runs automated tests on pull requests to ensure code quality before merging.

**Trigger:** Automatically runs on pull requests targeting the `master` branch

**Steps:**

1. **Checkout code** - Retrieves PR code
2. **Set up Python 3.11** - Configures environment
3. **Install dependencies** - Installs packages from `model_requirements.txt`
4. **Install dev/test dependencies** - Installs pytest, pytest-cov, and seaborn
5. **Install project as package** - Installs the project in editable mode (`pip install -e .`)
6. **Run Tests** - Executes pytest with verbose output on all tests in `./tests` directory

**Key Features:**
- **Pre-Merge Validation**: Ensures code quality before merging
- **Comprehensive Testing**: Runs all unit tests
- **Fast Feedback**: Provides quick feedback to developers

**Test Coverage:**
- Unit tests for data processing modules
- Unit tests for bias detection and mitigation
- Unit tests for model training and evaluation
- Integration tests where applicable

---

#### Send Email Notification (`send_email.yml`)

**Purpose:** Reusable workflow for sending email notifications about workflow status.

**Trigger:** Called by other workflows using `workflow_call`

**Inputs:**
- `status`: Workflow execution status (success/failure)
- `workflow_name`: Name of the workflow that triggered the notification

**Steps:**

1. **Send email via Gmail SMTP** - Executes Python script that:
   - Connects to Gmail SMTP server (smtp.gmail.com:587)
   - Constructs email with workflow information:
     - Workflow name
     - Status (success/failure)
     - Repository name
     - Commit SHA
     - Branch name
   - Sends email to configured recipients

**Key Features:**
- **Reusable**: Can be called by any workflow
- **Configurable**: Uses GitHub secrets for SMTP credentials
- **Informative**: Includes all relevant workflow context

**Configuration:**
- `SMTP_EMAIL`: Gmail address for sending notifications
- `SMTP_PASSWORD`: Gmail app password (stored in GitHub secrets)

---

### Pipeline Flow Diagram

```text
┌─────────────────┐
│  1. Load Data   │ (Manual Trigger)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Train Model  │ (Auto: After Load Data)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ 3. Generate          │ (Auto: After Training)
│    Predictions       │
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│ 4. Evaluate     │ (Auto: After Predictions)
│    Model        │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ 5. Bias Detection     │ (Auto: After Evaluation)
│    & Mitigation       │
└────────┬──────────────┘
         │
         ▼
┌─────────────────┐
│ 6. Model         │ (Auto: After Bias Pipeline)
│    Validation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. Model Manager│ (Auto: After Validation)
└─────────────────┘

┌─────────────────┐
│  PR Test Suite  │ (On Pull Requests)
└─────────────────┘

┌─────────────────┐
│ Send Email      │ (Called by all workflows)
│ Notification    │
└─────────────────┘
```

### Key Features of the Pipeline

1. **Sequential Execution**: Each workflow triggers the next only on success
2. **Failure Handling**: Pipeline stops on failure, preventing bad models from progressing
3. **Artifact Management**: All artifacts committed to `artifacts_bot` branch for versioning
4. **MLflow Integration**: Automatic MLflow server management for experiment tracking
5. **Email Notifications**: Automated notifications for every workflow execution
6. **Quality Gates**: Multiple validation checkpoints ensure model quality
7. **Bias Detection**: Comprehensive bias analysis before deployment
8. **Automated Versioning**: Model version management integrated into pipeline

### Required GitHub Secrets

The workflows require the following secrets to be configured in GitHub:

- `GCP_CREDENTIALS`: Google Cloud Platform service account JSON credentials
- `SMTP_EMAIL`: Gmail address for sending notifications
- `SMTP_PASSWORD`: Gmail app password for SMTP authentication
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions (for artifact access)

### Artifact Branch Strategy

All generated artifacts (models, reports, visualizations) are committed to the `artifacts_bot` branch rather than the main branch. This strategy:

- **Keeps main clean**: Main branch contains only source code
- **Version control**: All artifacts are versioned and tracked
- **Easy access**: Artifacts can be accessed via the dedicated branch
- **CI/CD friendly**: Doesn't trigger additional workflows from artifact commits

### Monitoring and Debugging

- **Workflow Logs**: Detailed logs available in GitHub Actions UI
- **Email Notifications**: Receive notifications for every workflow run
- **Artifact Inspection**: All artifacts available in `artifacts_bot` branch
- **MLflow Tracking**: Model metrics tracked in MLflow (when MLflow server is running)

## Data Pipeline Architecture & Components

This section provides a detailed breakdown of the data pipeline components and their functionality within the Apache Airflow DAG.

### Pipeline Overview

The `goodreads_recommendation_pipeline` DAG orchestrates a comprehensive data processing workflow that transforms raw Goodreads data into a machine learning-ready dataset. The pipeline follows MLOps best practices with data validation, cleaning, feature engineering, and normalization stages.

### Pipeline Components (DAG Tasks)

#### 1. **Data Reading & Validation**

- **Task ID:** `read_data_from_bigquery`
- **Type:** `BigQueryInsertJobOperator`

- **Purpose:** Extracts data from Google BigQuery source tables and performs initial record count validation

- **Functionality:**
  - Queries both books and interactions tables from the mystery/thriller/crime genre subset
  - Validates data availability and record counts
  - Establishes connection to GCP BigQuery using configured credentials
  - Provides baseline metrics for data quality assessment

#### 2. **Results Logging**

- **Task ID:** `log_bq_results`
- **Type:** `PythonOperator`

- **Purpose:** Processes and logs the results from the BigQuery data reading task

- **Functionality:**
  - Retrieves job results from the previous BigQuery operation
  - Logs detailed information about data availability
  - Provides visibility into data extraction success metrics
  - Enables monitoring of data pipeline health

#### 3. **Pre-Cleaning Data Validation**

- **Task ID:** `validate_data_quality`
- **Type:** `PythonOperator`

- **Purpose:** Performs comprehensive data quality checks on raw source data

- **Functionality:**
  - Validates table structure and schema compliance
  - Checks for required columns and data types
  - Identifies missing values and data range violations
  - Detects anomalies in rating distributions and user behavior patterns
  - Stops pipeline execution if critical data quality issues are found
  - Sends failure notifications via email for immediate alerting

- **Validation Logic:**
  - **Books Table (Source):**
    - **Critical Field Validation:** Ensures `book_id` and `title` are never null
    - **Business Rule:** These are essential identifiers that cannot be missing
    - **Zero Tolerance:** Any null values cause validation failure
  - **Interactions Table (Source):**
    - **Referential Integrity:** Ensures both `user_id` and `book_id` are present
    - **Relationship Validation:** These are foreign keys that must exist
    - **Zero Tolerance:** Missing identifiers would break user-book relationships

#### 4. **Data Cleaning**

- **Task ID:** `clean_data`
- **Type:** `PythonOperator`

- **Purpose:** Cleans and standardizes the raw Goodreads dataset

- **Functionality:**
  - Removes duplicate records and invalid entries
  - Standardizes text fields (titles, authors, descriptions)
  - Handles missing values using appropriate imputation strategies
  - Cleans and validates timestamp fields for reading dates
  - Removes outliers and invalid ratings
  - Creates cleaned tables in BigQuery for downstream processing

#### 5. **Post-Cleaning Validation**

- **Task ID:** `validate_cleaned_data`
- **Type:** `PythonOperator`

- **Purpose:** Validates data quality after cleaning operations

- **Functionality:**
  - Ensures cleaning process completed successfully
  - Verifies data integrity and consistency
  - Validates that cleaned data meets quality standards
  - Checks for any new issues introduced during cleaning
  - Provides confidence in data quality for feature engineering

- **Validation Logic:**
  - **Books Table (Cleaned):**
    - **Data Integrity:** Ensures cleaning process didn't introduce new nulls in `title`
    - **Range Validation:** Validates `publication_year` is reasonable (1000-2030)
    - **Business Rules:** `num_pages` must be positive and realistic (≤10,000 pages)
    - **ML Readiness:** Prevents unrealistic data from entering the ML pipeline
  - **Interactions Table (Cleaned):**
    - **Data Integrity Preservation:** Ensures cleaning didn't corrupt essential fields
    - **Rating Validation:** Validates ratings are within expected range (0-5)
    - **ML Pipeline Readiness:** Ensures data is suitable for recommendation algorithms

#### 6. **Feature Engineering**

- **Task ID:** `feature_engg_data`
- **Type:** `PythonOperator`

- **Purpose:** Creates machine learning features from cleaned data

- **Functionality:**
  - **Book-level Features:**
    - Average reading time per book across all readers
    - Book popularity metrics (total ratings, average rating)
    - Genre and publication year features
    - Text-based features from book descriptions
  - **User-level Features:**
    - User reading patterns and preferences
    - Reading speed and completion rates
    - Genre preferences and rating patterns
  - **Interaction Features:**
    - User-book interaction history
    - Temporal features (reading dates, seasonal patterns)
    - Rating patterns and review sentiment

#### 7. **Data Normalization**

- **Task ID:** `normalize_data`
- **Type:** `PythonOperator`

- **Purpose:** Normalizes features for machine learning model consumption

- **Functionality:**
  - Applies appropriate scaling techniques (Min-Max, Z-score normalization)
  - Handles categorical variable encoding
  - Ensures feature distributions are suitable for ML algorithms
  - Creates final normalized dataset ready for model training
  - Maintains data consistency across different feature types

#### 8. **Promote Staging**

- **Task ID:** `promote_staging_tables`  
- **Type:** `PythonOperator`  

- **Purpose:** Promotes data from staging tables to final production tables.

- **Functionality:**  
  - Each preceding task reads from and writes to staging tables to ensure data integrity during intermediate steps.  
  - Once all validations and tests pass, this task promotes the data by moving it from staging tables to the final tables.  
  - After promotion, the staging tables are deleted to maintain a clean workspace.  
  - The finalized tables are then made available for downstream modeling tasks.

---

#### 9. **Data Versioning**

- **Task ID:** `data_versioning_task`  
- **Type:** `PythonOperator`  

- **Purpose:** Implements data version control using DVC.

- **Functionality:**  
  - The final features table is versioned using **Data Version Control (DVC)** to track changes over time.  
  - DVC is integrated with **Google Cloud Storage (GCS)** to enable persistent and scalable storage of versioned datasets.  
  - This ensures reproducibility, auditability, and consistency across different pipeline runs.

---

#### 10. **Train, Test, and Validation Split**

- **Task ID:** `train_test_split_task`  
- **Type:** `PythonOperator`  

- **Purpose:** Splits the processed data into training, testing, and validation sets for model development.

- **Functionality:**  
  - The final features table is partitioned into:  
    - **Training Set (70%)** — used to train the model.  
    - **Validation Set (15%)** — used for hyperparameter tuning and model selection.  
    - **Test Set (15%)** — used for final evaluation and performance assessment.  
  - This ensures a robust evaluation framework and prevents data leakage between training and testing phases.


### Pipeline Flow & Dependencies

```text
start → read_data_from_bigquery → log_bq_results → validate_data_quality 
    → clean_data → validate_cleaned_data → feature_engg_data → normalize_data → end
```

### Error Handling & Monitoring

- **Email Notifications:** Automatic failure and success notifications sent to configured email addresses
- **Retry Logic:** Configurable retry attempts with exponential backoff
- **Logging:** Comprehensive logging at each stage for debugging and monitoring
- **Data Quality Gates:** Pipeline stops if critical data quality issues are detected
- **BigQuery Integration:** Seamless integration with Google Cloud Platform for scalable data processing

### Data Quality Validation Framework

The pipeline implements a comprehensive data quality validation system using BigQuery SQL queries that operates at two critical stages: **pre-cleaning** and **post-cleaning**. This ensures data integrity throughout the entire processing workflow.

#### **Validation Architecture**

The validation system uses the `AnomalyDetection` class with the following key components:

- **BigQuery Integration:** Direct SQL query execution for scalable validation
- **Progressive Validation:** Different validation rules for source vs. cleaned data
- **Zero Tolerance Policy:** All validations use `max_allowed: 0` - any violations stop the pipeline
- **Comprehensive Logging:** Detailed validation results logged for monitoring
- **Email Alerts:** Automatic failure notifications sent to stakeholders

#### **Pre-Cleaning Validation (Source Data)**

**Purpose:** Validates raw data before any processing to ensure basic data integrity.

**Books Table Validation:**

```sql
-- Critical field validation
SELECT COUNT(*) as null_count
FROM `project.books.goodreads_books_mystery_thriller_crime`
WHERE book_id IS NULL OR title IS NULL
```

**Interactions Table Validation:**

```sql
-- Referential integrity validation
SELECT COUNT(*) as null_count
FROM `project.books.goodreads_interactions_mystery_thriller_crime`
WHERE user_id IS NULL OR book_id IS NULL
```

**Validation Logic:**

- **Critical Field Validation:** Ensures `book_id` and `title` are never null
- **Referential Integrity:** Ensures both `user_id` and `book_id` are present
- **Business Rule:** These are essential identifiers that cannot be missing
- **Zero Tolerance:** Any null values cause validation failure

#### **Post-Cleaning Validation (Cleaned Data)**

**Purpose:** Validates cleaned data to ensure business rules and ML readiness requirements are met.

**Books Table Validation:**

```sql
-- Data integrity and range validation
SELECT COUNT(*) as invalid_count
FROM `project.books.goodreads_books_cleaned_staging`
WHERE title IS NULL 
   OR publication_year IS NULL 
   OR publication_year <= 0 
   OR num_pages IS NULL 
   OR num_pages <= 0
```

**Interactions Table Validation:**

```sql
-- Data integrity and rating validation
SELECT COUNT(*) as invalid_count
FROM `project.books.goodreads_interactions_cleaned_staging`
WHERE user_id IS NULL 
   OR book_id IS NULL 
   OR rating < 0 
   OR rating > 5
```

**Validation Logic:**

- **Data Integrity:** Ensures cleaning process didn't introduce new nulls
- **Range Validation:** Validates `publication_year` is reasonable (1000-2030)
- **Business Rules:** `num_pages` must be positive and realistic (≤10,000 pages)
- **Rating Validation:** Validates ratings are within expected range (0-5)
- **ML Readiness:** Ensures data is suitable for recommendation algorithms

#### **Key Design Principles**

1. **Progressive Validation:** Pre-cleaning focuses on critical fields, post-cleaning adds business rule validation
2. **Zero Tolerance:** All validations use `max_allowed: 0` - any violations stop the pipeline
3. **Business Logic:** Range checks ensure data makes business sense
4. **ML Pipeline Readiness:** Post-cleaning validation ensures data is suitable for machine learning
5. **Comprehensive Error Handling:** Detailed logging and email notifications for failures

#### **Pipeline Integration**

The validation system integrates with the Airflow DAG at two critical points:

1. **Pre-cleaning** (`validate_data_quality` task): Validates source data before any processing
2. **Post-cleaning** (`validate_cleaned_data` task): Validates cleaned data before feature engineering

This ensures data quality gates at both ends of the cleaning process, preventing bad data from propagating through the ML pipeline and ensuring the final dataset meets the requirements for recommendation system training.

### **Bottleneck Identification**

Initially, the data was fetched as JSON files and processed locally. However, as the dataset was much large, the pipeline became increasingly slow and resource-intensive. To improve scalability and performance, we restructured the pipeline to store data in Google Cloud Storage and load it into BigQuery tables. This allows us to perform transformations and processing directly through SQL queries in BigQuery, significantly enhancing the efficiency of the data pipeline.

### Configuration & Environment

- **Airflow Configuration:** Managed through `config/airflow.cfg`
- **GCP Credentials:** Requires `gcp_credentials.json` in the config directory
- **Connection Management:** Uses `goodreads_conn` for BigQuery connectivity
- **Environment Variables:** `AIRFLOW_HOME` and `GOOGLE_APPLICATION_CREDENTIALS` configuration

## Folder Structure

| Folder/File                    | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `config/`                     | Airflow configuration files, DAG definitions, and database files for pipeline orchestration |
| `config/dags/`                | Apache Airflow DAG files defining the data pipeline workflow and task dependencies |
| `datapipeline/`               | Core data processing modules including data cleaning, feature engineering, and validation scripts |
| `datapipeline/data/`          | Data storage directories for raw, processed datasets and analysis notebooks |
| `datapipeline/data/raw/`      | Raw Goodreads dataset storage before any processing or cleaning operations |
| `datapipeline/data/processed/`| Cleaned and processed datasets with schema definitions for ML pipeline consumption |
| `datapipeline/data/notebooks/`| Jupyter notebooks for exploratory data analysis, bias analysis, and model prototyping |
| `datapipeline/scripts/`       | Main data processing scripts including cleaning, feature engineering, normalization, and anomaly detection |
| `datapipeline/tests/`         | Unit tests for data processing components to ensure code quality and reliability |
| `docs/`                       | Project documentation including scope definitions and technical specifications |
| `features.md`                 | Feature documentation describing implemented functionality and capabilities |
| `requirements.txt`            | Python package dependencies and version specifications for environment setup |
| `setup.py`                    | Python package configuration for installation and distribution management |
| `README.md`                   | Project documentation with setup instructions, architecture overview, and usage guidelines |

### Core ML Pipeline Modules (`src/`)

The `src/` directory groups together the reusable services that orchestrate model training, fairness analysis, and deployment. Each module is designed to be imported by Airflow DAGs, CLI utilities, or notebooks:

| Module | Purpose |
|--------|---------|
| `src/__init__.py` | Exposes the public package surface so DAGs/notebooks can import shared helpers without deep relative paths. |
| `src/load_data.py` | Bootstraps BigQuery credentials and returns the curated train split as either BigFrames or pandas DataFrames. |
| `src/df_model_training.py` | Lightweight, in-memory training stub for experimentation and CI smoke tests using pandas data. |
| `src/bq_model_training.py` | Production BigQuery ML training workflow with concurrency handling, MLflow logging, and evaluation hooks. |
| `src/generate_prediction_tables.py` | Builds bias-ready prediction tables (features + inferred slices) for every trained model. |
| [`src/bias_detection.py`](src/bias_detection.py) | Computes slice-aware performance metrics, disparity summaries, and mitigation recommendations. |
| [`src/bias_mitigation.py`](src/bias_mitigation.py) | Implements shrinkage, threshold adjustment, and re-weighting strategies to reduce detected bias. |
| [`src/bias_pipeline.py`](src/bias_pipeline.py) | End-to-end orchestrator that stitches detection, mitigation, visualization, and reporting steps. |
| [`src/bias_visualization.py`](src/bias_visualization.py) | Generates fairness scorecards, disparity heatmaps, and slice comparison plots for reports. |
| `src/model_selector.py` | Balances validation accuracy with fairness scores to pick the best candidate model. |
| `src/model_manager.py` | Compares new Vertex AI versions to the current default and promotes/rolls back based on RMSE deltas. |
| [`src/model_evaluation_pipeline.py`](src/model_evaluation_pipeline.py) | Logs MAE/RMSE, computes SHAP-based feature importance, and exports evaluation artifacts. |
| [`src/model_sensitivity_analysis.py`](src/model_sensitivity_analysis.py) | Provides SHAP helpers and visualizations to explain feature impact across models. |
| `src/model_validation.py` | Runs ML.EVALUATE on train/val/test splits, enforces RMSE thresholds, and persists JSON reports. |
| `src/register_bqml_models.py` | Uploads the latest BQML artifacts into the Vertex AI Model Registry with version control. |

### Bias Reporting Artifacts (`docs/bias_reports/`)

Bias analysis is treated as a first-class deliverable. Everything the bias pipeline emits is organized under `docs/bias_reports/`, making it easy for reviewers to trace fairness conclusions without re-running notebooks:

| Artifact | Produced By | Detailed Contents & Intended Use |
|----------|-------------|----------------------------------|
| `*_detection_report.json` | `bias_detection.py` | For each model/dataset pair we persist the exact slice metrics (MAE, RMSE, counts, mean error), the disparity summary per dimension, the list of high-risk slices, and the auto-generated recommendations. Example: [`boosted_tree_regressor_detection_report.json`](docs/bias_reports/boosted_tree_regressor_detection_report.json). These JSONs are ingested by dashboards and also used later when comparing mitigation effectiveness. |
| `*_mitigation_report.json` | `bias_mitigation.py` | Captures before/after metrics for the specific technique that ran (shrinkage, threshold adjustment, re-weighting). Includes parameter choices (e.g., lambda, threshold deltas), improvement percentages, and the BigQuery table where debiased predictions were written, so QA can reproduce results. |
| `*_comprehensive_audit.json` | `bias_pipeline.py` | Serves as the executive summary for a full run: metadata about the audit, detection highlights, which mitigation steps executed, validation re-checks, and an executive summary block with pass/mitigated/needs-attention status. Example: [`boosted_tree_regressor_comprehensive_audit.json`](docs/bias_reports/boosted_tree_regressor_comprehensive_audit.json). This is what we attach to compliance reviews. |
| `bias_pipeline_logs/` | `bias_pipeline.py` CLI / Airflow task | Raw stdout/stdio logs captured with timestamps, showing every step, warning, and recommendation printed during the run. These logs are critical when investigating why a mitigation was skipped or why BigQuery jobs failed. |
| `visualizations/` | `bias_visualization.py` | Collection of PNGs referenced in reports: fairness scorecard gauges, disparity heatmaps, per-dimension bar charts, before/after comparisons, etc. Each filename embeds the model name and slice to simplify embedding in slide decks. |
| `model_selection/` | `model_selector.py` | Stores [`model_selection_report.json`](docs/bias_reports/model_selection_report.json) (combined performance + fairness scoring), along with charts such as `model_comparison.png` so stakeholders can visually inspect the trade-offs the selector considered. |
| `model_selection_report.json` (root copy) | `model_selector.py` | Latest snapshot of the selection decision for downstream automation (e.g., `model_manager.py` and `model_validation.py` read this file to know which table to promote or validate). |

Together, these artifacts create an auditable trail from raw metrics → mitigation decisions → final deployment approvals. Whenever a regression is suspected we can diff the JSON reports, compare visualization sets, and replay the exact mitigation query using the table paths stored in the reports.

### Model Analysis Artifacts (`docs/model_analysis/`)

Complementing the bias reports, `docs/model_analysis/` captures performance-centric evaluations so data scientists can answer “how accurate?” and “why?” in one place:

| Subfolder / File | Produced By | What It Contains & Why It Matters |
|------------------|-------------|-----------------------------------|
| `evaluation/` | `model_evaluation_pipeline.py` | JSON reports such as [`boosted_tree_regressor_evaluation_report.json`](docs/model_analysis/evaluation/boosted_tree_regressor_evaluation_report.json) that store run metadata, MAE/RMSE/correlation statistics, accuracy-within-Δ buckets, and pointers to any sensitivity analysis outputs. These files give product owners a single source of truth for offline performance. |
| `sensitivity/` | `model_sensitivity_analysis.py` | SHAP analysis artifacts (JSON summaries plus PNG charts referenced in docs). Each JSON (e.g., [`test-model_feature_importance.json`](docs/model_analysis/sensitivity/test-model_feature_importance.json)) lists feature importance scores, categorical encodings, sample sizes, and file paths for the generated plots, enabling explainability reviews without rerunning SHAP. |

By pairing `docs/bias_reports/` with `docs/model_analysis/`, we maintain a clear separation: bias artifacts answer “is it fair?”, while model analysis artifacts answer “is it accurate and interpretable?”. Both folders are versioned so we can compare historical runs.

#### What do the evaluation JSON files look like?

Every JSON inside `docs/model_analysis/evaluation/` follows a consistent schema:

```json
{
  "model_name": "boosted_tree_regressor",
  "timestamp": "2025-01-15T04:27:13.829410",
  "predictions_table": "project.books.boosted_tree_rating_predictions",
  "performance_metrics": {
    "num_predictions": 12456,
    "mae": 0.6123,
    "rmse": 0.8124,
    "r_squared": 0.54,
    "accuracy_within_0_5_pct": 62.1,
    "accuracy_within_1_0_pct": 88.8,
    "accuracy_within_1_5_pct": 97.4,
    "mean_predicted": 3.84,
    "mean_actual": 3.79,
    "std_error": 0.41
  },
  "sensitivity_analysis": {
    "artifact_path": "../docs/model_analysis/sensitivity/boosted_tree_regressor_feature_importance.json",
    "top_features": [
      {"feature": "book_popularity_normalized", "importance": 0.142},
      {"feature": "user_activity_count", "importance": 0.117}
    ]
  }
}
```

- `performance_metrics` is exactly what gets logged to MLflow; keeping it in JSON allows dashboards or audits to ingest it directly.
- `sensitivity_analysis` is optional, but when present it points to the SHAP JSON so reviewers can jump straight from accuracy numbers to feature explanations.
- The `predictions_table` reference lets anyone regenerate metrics or spot-check rows in BigQuery.

#### What do the sensitivity JSON files look like?

Every JSON inside `docs/model_analysis/sensitivity/` follows a consistent schema for feature importance analysis:

```json
{
  "model_name": "test-model",
  "timestamp": "2025-11-17T04:53:33.161052",
  "sample_size": 100,
  "feature_importance": [
    {
      "feature": "num_genres",
      "importance": 0.225
    },
    {
      "feature": "book_popularity_normalized",
      "importance": 0.125
    }
  ],
  "categorical_mappings": {}
}
```

**Schema Explanation:**

- `model_name` - Identifies which model was analyzed (e.g., `boosted_tree_regressor`, `matrix_factorization`)
- `timestamp` - ISO format timestamp of when the analysis was run
- `sample_size` - Number of samples used for SHAP analysis (typically 100-1000 for computational efficiency)
- `feature_importance` - Array of feature importance scores computed using SHAP values:
  - Each entry contains `feature` name and `importance` score (absolute mean SHAP value)
  - Features are sorted by importance (highest first)
  - Importance scores indicate how much each feature contributes to predictions on average
- `categorical_mappings` - Dictionary mapping categorical feature names to their encoded values (used for interpretability)

**Usage:**

- These files enable explainability reviews without rerunning computationally expensive SHAP analysis
- Feature importance scores help identify which features drive model predictions
- Comparison across models reveals which features are consistently important
- The JSON format allows easy integration with dashboards and reporting tools

**Example Files:**

- [`test-model_feature_importance.json`](docs/model_analysis/sensitivity/test-model_feature_importance.json) - Sample sensitivity analysis output
- Model-specific files follow the pattern: `{model_name}_feature_importance.json`

### Hyperparameter Sensitivity Analysis

Hyperparameter tuning is critical for optimizing model performance. We conducted comprehensive hyperparameter sensitivity analysis for both our primary models to understand how different parameter settings affect model performance.

#### Boosted Tree Regressor Hyperparameter Sensitivity

The Boosted Tree model's performance is sensitive to several key hyperparameters. We analyzed the impact of:

1. **Number of Trees** - Controls model complexity and overfitting risk
2. **Tree Depth** - Determines the depth of individual trees in the ensemble
3. **Subsample Ratio** - Controls the fraction of training data used for each tree

<div>
  <p align="center">
    <img src="assets/BT_num_trees.png" alt="Boosted Tree Number of Trees Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Number of Trees Hyperparameter Sensitivity</em>
  </p>

  <p align="center">
    <img src="assets/BT_tree_depth.png" alt="Boosted Tree Depth Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Tree Depth Hyperparameter Sensitivity</em>
  </p>

  <p align="center">
    <img src="assets/BT_subsample.png" alt="Boosted Tree Subsample Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Subsample Ratio Hyperparameter Sensitivity</em>
  </p>

  <p align="center">
    <img src="assets/BT_training.png" alt="Boosted Tree Training Progress" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Training Progress and Convergence</em>
  </p>
</div>

**Key Findings for Boosted Tree:**

- **Optimal Number of Trees**: Found to be in the range that balances bias and variance without overfitting
- **Tree Depth**: Moderate depth provides best generalization performance
- **Subsample Ratio**: Lower subsample ratios help prevent overfitting while maintaining model performance
- **Training Convergence**: Model shows stable convergence with selected hyperparameters

#### Matrix Factorization Hyperparameter Sensitivity

The Matrix Factorization model's performance depends on factorization parameters and regularization:

1. **Number of Factors** - Controls the dimensionality of the latent feature space
2. **L2 Regularization** - Prevents overfitting by penalizing large parameter values
3. **Number of Iterations** - Determines training duration and convergence

<div>
  <p align="center">
    <img src="assets/MF_num_factors.png" alt="Matrix Factorization Number of Factors Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Number of Factors Hyperparameter Sensitivity</em>
  </p>

  <p align="center">
    <img src="assets/MF_l2_reg.png" alt="Matrix Factorization L2 Regularization Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>L2 Regularization Hyperparameter Sensitivity</em>
  </p>

  <p align="center">
    <img src="assets/MF_iterations.png" alt="Matrix Factorization Iterations Sensitivity" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Number of Iterations Hyperparameter Sensitivity</em>
  </p>
</div>

**Key Findings for Matrix Factorization:**

- **Optimal Number of Factors**: Moderate factor count provides best balance between model expressiveness and generalization
- **L2 Regularization**: Appropriate regularization strength prevents overfitting while maintaining recommendation quality
- **Iterations**: Model converges efficiently with optimal iteration count
- **Performance Trade-offs**: Clear trade-offs between model complexity and prediction accuracy identified

**Hyperparameter Tuning Methodology:**

- **Grid Search**: Systematic exploration of hyperparameter space
- **Cross-Validation**: Used to evaluate hyperparameter combinations
- **Performance Metrics**: RMSE and MAE used as primary evaluation metrics
- **Computational Efficiency**: Balanced thoroughness with training time constraints

These sensitivity analyses informed our final hyperparameter selections, ensuring optimal model performance while maintaining computational efficiency.

## Bias Detection and Mitigation System

Our recommendation system includes a comprehensive bias detection and mitigation framework that ensures fair and equitable recommendations across multiple demographic and feature dimensions. This system is integrated into the ML pipeline and automatically audits models for potential bias before deployment.

### Overview

The bias detection and mitigation system operates as a multi-stage pipeline that:

1. **Detects Bias** - Analyzes model predictions across multiple demographic slices to identify performance disparities
2. **Applies Mitigation** - Implements appropriate mitigation techniques to reduce detected bias
3. **Validates Effectiveness** - Re-evaluates models post-mitigation to ensure improvements
4. **Generates Reports** - Creates comprehensive audit reports with visualizations for stakeholders

### Bias Dimensions Analyzed

The system analyzes bias across **8 key dimensions** to ensure fair recommendations:

1. **Book Popularity** - High/Medium/Low popularity groups (based on `book_popularity_normalized`)
   - Identifies whether the model performs differently for popular vs. niche books
   - Ensures lesser-known books receive fair rating predictions

2. **Book Length** - Categories based on `book_length_category` (Short/Medium/Long)
   - Detects bias against books of different lengths
   - Ensures short and long books are evaluated fairly

3. **Book Era** - Publication era groups (`book_era`)
   - Identifies temporal bias (e.g., modern vs. classic books)
   - Ensures historical and contemporary works receive equitable treatment

4. **Genre Diversity** - Multi-genre vs. single-genre books
   - Detects bias based on genre complexity
   - Ensures diverse and focused works are treated fairly

5. **User Activity Level** - High/Medium/Low activity user groups (based on `user_activity_count`)
   - Identifies whether model performance differs for active vs. casual users
   - Ensures all user segments receive quality recommendations

6. **Reading Pace** - Fast/Medium/Slow reader categories (`reading_pace_category`)
   - Detects bias related to reading speed preferences
   - Ensures recommendations are fair across reading styles

7. **Author Gender** - Male/Female author groups (`author_gender_group`)
   - Identifies potential gender bias in recommendations
   - Ensures equitable treatment of works by authors of all genders

8. **Rating Range** - High (4-5)/Medium (3-4)/Low (1-3) actual rating groups
   - Detects bias in how the model handles different rating ranges
   - Ensures consistent performance across all rating levels

### Bias Detection Process

The detection system uses a **slice-based analysis** approach:

#### 1. Slice Metrics Computation

For each dimension, the system:

- **Splits predictions** into groups based on the dimension (e.g., High/Medium/Low popularity)
- **Computes performance metrics** for each slice:

  - **MAE (Mean Absolute Error)** - Average prediction error per slice
  - **RMSE (Root Mean Squared Error)** - Penalizes larger errors
  - **Mean Predicted vs. Actual** - Identifies systematic over/under-prediction
  - **Error Distribution** - Mean and standard deviation of errors

#### 2. Disparity Analysis

The system identifies bias by analyzing:

- **Coefficient of Variation (CV)** - Measures relative variability of MAE across slices
  - High CV (>0.2) indicates significant disparity
- **Range Analysis** - Compares best vs. worst performing slices
- **Statistical Significance** - Identifies slices with statistically different performance

#### 3. High-Risk Slice Identification

Slices are flagged as high-risk if they:

- Have MAE significantly above the global average
- Show systematic over/under-prediction patterns
- Have high error variance relative to other slices

### Mitigation Techniques

When bias is detected, the system applies appropriate mitigation techniques:

#### 1. Prediction Shrinkage (Primary Method)

**How it works:**

- Pulls group-specific prediction means toward the global mean
- Formula: `prediction_debiased = original_prediction - λ * (group_mean - global_mean)`
- Lambda (λ) parameter controls shrinkage strength (default: 0.3-0.5)

**When used:**

- Most effective for Book Era bias (temporal disparities)
- Applied directly to predictions without retraining
- Fast and production-ready

**Benefits:**

- Reduces extreme group differences
- Preserves overall model accuracy
- Maintains relative ordering within groups

#### 2. Threshold Adjustment

**How it works:**

- Applies different decision thresholds per group
- Adjusts prediction boundaries to balance performance

**When used:**

- For dimensions with clear threshold-based disparities
- When shrinkage alone is insufficient

#### 3. Re-weighting (Training-time)

**How it works:**

- Adjusts training sample weights to balance group representation
- Uses inverse propensity or balanced weighting strategies

**When used:**

- For severe bias requiring retraining
- When multiple dimensions show significant disparities

### Bias Audit Pipeline Workflow

The complete bias audit follows this workflow:

```text
1. Bias Detection
   ├── Slice predictions by dimension
   ├── Compute metrics per slice
   ├── Analyze disparities
   └── Generate detection report

2. Visualization Generation
   ├── Fairness scorecards
   ├── Disparity heatmaps
   ├── Per-dimension bar charts
   └── Before/after comparisons

3. Mitigation Application (if bias detected)
   ├── Select appropriate technique
   ├── Apply mitigation to predictions
   ├── Generate mitigated predictions table
   └── Create mitigation report

4. Validation
   ├── Re-run detection on mitigated predictions
   ├── Compare before/after metrics
   ├── Verify improvement thresholds
   └── Generate validation report

5. Comprehensive Reporting
   ├── Executive summary
   ├── Detailed metrics
   ├── Mitigation effectiveness
   └── Deployment recommendations
```

### Integration with ML Pipeline

The bias system integrates seamlessly with the ML pipeline:

- **Automatic Auditing** - Models are automatically audited after training
- **Pre-Deployment Gates** - Models with unacceptable bias are flagged before deployment
- **Version Tracking** - All bias reports are versioned alongside model versions
- **BigQuery Integration** - Bias metrics stored in BigQuery for historical analysis

### Bias Reporting Artifacts

All bias analysis results are stored in `docs/bias_reports/`:

#### Detection Reports (`*_detection_report.json`)

- Complete slice metrics for all dimensions
- Disparity analysis summaries
- High-risk slice identification
- Automated recommendations

#### Mitigation Reports (`*_mitigation_report.json`)

- Before/after metrics comparison
- Mitigation technique details
- Improvement percentages
- Output table references

#### Comprehensive Audit Reports (`*_comprehensive_audit.json`)

- Executive summary with pass/mitigate/needs-attention status
- Detection highlights
- Mitigation steps executed
- Validation results
- Deployment readiness assessment

#### Visualizations (`visualizations/`)

- Fairness scorecard gauges
- Disparity heatmaps
- Per-dimension comparison charts
- Before/after mitigation visualizations

### Usage Example

```python
from src.bias_pipeline import BiasAuditPipeline

# Initialize pipeline
pipeline = BiasAuditPipeline(project_id="your-project-id")

# Run full audit
results = pipeline.run_full_audit(
    model_name="boosted_tree_regressor",
    predictions_table="project.books.boosted_tree_rating_predictions",
    apply_mitigation=True,
    mitigation_techniques=['prediction_shrinkage'],
    generate_visualizations=True
)

# Access results
print(f"Bias detected: {len(results['detection_report'].disparity_analysis['detailed_disparities'])} dimensions")
print(f"Mitigation applied: {len(results['mitigation_results'])} techniques")
print(f"Production table: {results['debiased_table']}")
```

**Related Source Files:**

- [`src/bias_pipeline.py`](src/bias_pipeline.py) - Main bias audit pipeline orchestrator
- [`src/bias_detection.py`](src/bias_detection.py) - Bias detection module
- [`src/bias_mitigation.py`](src/bias_mitigation.py) - Bias mitigation techniques
- [`src/bias_visualization.py`](src/bias_visualization.py) - Visualization generation

### Key Design Principles

1. **Comprehensive Coverage** - Analyzes multiple dimensions simultaneously
2. **Automated Detection** - No manual intervention required
3. **Actionable Mitigation** - Provides concrete techniques to reduce bias
4. **Validation-First** - Always validates mitigation effectiveness
5. **Audit Trail** - Complete documentation for compliance and review
6. **Production-Ready** - Generates tables ready for deployment

### Fairness Metrics

The system uses several fairness metrics:

- **Equity Index** - Overall fairness score combining variance reduction and mean alignment
- **MAE Coefficient of Variation** - Relative variability across slices
- **Disparity Ratio** - Worst-to-best slice performance ratio
- **Improvement Percentage** - Pre/post-mitigation improvement metrics

### Best Practices

1. **Regular Auditing** - Audit models after each training run
2. **Threshold Setting** - Define acceptable bias thresholds per use case
3. **Mitigation Validation** - Always validate mitigation effectiveness
4. **Historical Tracking** - Compare bias metrics across model versions
5. **Stakeholder Review** - Share comprehensive reports with stakeholders

This bias detection and mitigation system ensures that our recommendation models provide fair, equitable, and high-quality recommendations to all users, regardless of book characteristics, user behavior patterns, or demographic factors.

### Bias Reports and Visualizations

The bias detection and mitigation pipeline generates comprehensive reports and visualizations that provide insights into model fairness across all analyzed dimensions. These artifacts are essential for understanding bias patterns, validating mitigation effectiveness, and making informed deployment decisions.

#### Fairness Scorecards

Fairness scorecards provide an at-a-glance view of model fairness across all dimensions. Each scorecard shows:

- Overall fairness score
- Per-dimension fairness indicators
- High-risk slice identification
- Quick pass/fail status

<div>
  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_fairness_scorecard.png" alt="Boosted Tree Regressor Fairness Scorecard" style="max-width:800px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Boosted Tree Regressor Fairness Scorecard</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/matrix_factorization_fairness_scorecard.png" alt="Matrix Factorization Fairness Scorecard" style="max-width:800px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Matrix Factorization Fairness Scorecard</em>
  </p>
</div>

#### Disparity Heatmaps

Disparity heatmaps visualize performance disparities across all dimensions and slices, making it easy to identify patterns and outliers:

<div>
  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_disparity_heatmap.png" alt="Boosted Tree Regressor Disparity Heatmap" style="max-width:800px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Boosted Tree Regressor Disparity Heatmap</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/matrix_factorization_disparity_heatmap.png" alt="Matrix Factorization Disparity Heatmap" style="max-width:800px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Matrix Factorization Disparity Heatmap</em>
  </p>
</div>

#### Per-Dimension MAE Comparisons

Detailed per-dimension comparisons show how model performance varies across different slices within each dimension. These visualizations help identify specific areas where bias mitigation is needed.

**Boosted Tree Regressor - Dimension Comparisons:**

<div>
  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_Book_Era_mae_comparison.png" alt="Boosted Tree Book Era MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Book Era Dimension - MAE Comparison</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_Popularity_mae_comparison.png" alt="Boosted Tree Popularity MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Popularity Dimension - MAE Comparison</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_Author_Gender_mae_comparison.png" alt="Boosted Tree Author Gender MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Author Gender Dimension - MAE Comparison</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/boosted_tree_regressor_User_Activity_mae_comparison.png" alt="Boosted Tree User Activity MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>User Activity Dimension - MAE Comparison</em>
  </p>
</div>

**Matrix Factorization - Dimension Comparisons:**

<div>
  <p align="center">
    <img src="docs/bias_reports/visualizations/matrix_factorization_Book_Era_mae_comparison.png" alt="Matrix Factorization Book Era MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Book Era Dimension - MAE Comparison</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/matrix_factorization_Popularity_mae_comparison.png" alt="Matrix Factorization Popularity MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Popularity Dimension - MAE Comparison</em>
  </p>

  <p align="center">
    <img src="docs/bias_reports/visualizations/matrix_factorization_Author_Gender_mae_comparison.png" alt="Matrix Factorization Author Gender MAE Comparison" style="max-width:700px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Author Gender Dimension - MAE Comparison</em>
  </p>
</div>

**Additional Dimension Visualizations:**

All 8 dimensions are visualized for both models. Additional comparisons include:
- Book Length comparisons
- Genre Diversity comparisons
- Reading Pace comparisons
- Rating Range comparisons

Visualization files are stored in [`docs/bias_reports/visualizations/`](docs/bias_reports/visualizations/) and follow the naming pattern: `{model_name}_{dimension}_mae_comparison.png`

#### Model Selection Comparison

The model selection process compares multiple candidate models based on both performance metrics and fairness scores. The comparison chart visualizes the trade-offs between accuracy and fairness:

<div>
  <p align="center">
    <img src="docs/bias_reports/model_selection/model_comparison.png" alt="Model Selection Comparison Chart" style="max-width:800px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Model Selection Comparison - Performance vs. Fairness Trade-offs</em>
  </p>
</div>

**Model Selection Criteria:**

- **Performance Metrics**: RMSE, MAE, and R² scores
- **Fairness Scores**: Equity indices across all dimensions
- **Weighted Scoring**: Configurable weights for performance vs. fairness
- **Threshold Enforcement**: Minimum fairness requirements

The selected model balances accuracy with fairness, ensuring both high-quality recommendations and equitable treatment across all user and book segments.

**Model Selection Report:**

The detailed selection decision is documented in [`model_selection_report.json`](docs/bias_reports/model_selection_report.json), which includes:
- Candidate model scores
- Selection rationale
- Performance and fairness trade-offs
- Final model recommendation

#### Bias Pipeline Logs

Bias pipeline execution logs provide detailed audit trails of every bias detection and mitigation run. These logs are essential for:

- **Debugging**: Investigating why mitigations were skipped or failed
- **Reproducibility**: Understanding exact parameter choices and execution paths
- **Compliance**: Providing audit trails for regulatory reviews
- **Monitoring**: Tracking bias metrics over time

**Log File Structure:**

Logs are stored in [`docs/bias_reports/bias_pipeline_logs/`](docs/bias_reports/bias_pipeline_logs/) with timestamped filenames:
- Format: `pipeline_YYYYMMDD_HHMMSS.log`
- Example: `pipeline_20251112_220818.log`

**Log Contents Include:**

- Pipeline execution start/end times
- Model names and prediction table references
- Bias detection results per dimension
- Mitigation technique selections and parameters
- Validation results and improvement metrics
- Warnings and error messages
- Final recommendations and deployment status

**Example Log Files:**

- [`pipeline_20251112_220818.log`](docs/bias_reports/bias_pipeline_logs/pipeline_20251112_220818.log) - Sample pipeline execution log
- [`pipeline_20251113_174321.log`](docs/bias_reports/bias_pipeline_logs/pipeline_20251113_174321.log) - Additional execution log

**Accessing Logs:**

Logs can be viewed directly or parsed programmatically to extract specific metrics, track trends, or generate summary reports. They complement the JSON reports by providing the full execution context.

#### Report Artifacts Summary

All bias analysis artifacts are organized in [`docs/bias_reports/`](docs/bias_reports/):

```
docs/bias_reports/
├── *_detection_report.json          # Detailed bias detection results
├── *_comprehensive_audit.json        # Executive summary reports
├── model_selection_report.json       # Model selection decision
├── visualizations/                   # All visualization PNGs
│   ├── *_fairness_scorecard.png
│   ├── *_disparity_heatmap.png
│   └── *_*_mae_comparison.png
├── model_selection/                  # Model comparison charts
│   └── model_comparison.png
└── bias_pipeline_logs/               # Execution logs
    └── pipeline_*.log
```

These artifacts together provide a complete picture of model fairness, enabling data scientists, product managers, and compliance officers to make informed decisions about model deployment.

## Data Analysis & Insights

This section highlights key findings from our exploratory data analysis and bias detection notebooks that inform our recommendation system design.

### Raw Data Analysis

**Notebook:** [`raw_analysis.ipynb`](datapipeline/data/notebooks/raw_analysis.ipynb)

**Purpose:** BigQuery data exploration to inspect raw data schema, null patterns, and data distributions.

**Analysis Performed:**

- **BigQuery Connection:** Connected to `recommendation-system-475301` project and queried `goodreads_books_mystery_thriller_crime` table
- **Data Sampling:** Retrieved 10,000 rows with 7 columns (book_id, title, authors, average_rating, ratings_count, popular_shelves, description)
- **Data Quality Assessment:**
  - Zero missing values across all columns
  - 9,599 unique titles out of 10,000 books
  - 235 unique average rating values
  - 1,353 unique ratings count values
  - 7,367 unique descriptions
- **JSON Field Analysis:** Examined structured data in authors and popular_shelves columns
- **Statistical Summary:** Generated descriptive statistics and data type analysis

**Visualizations Generated:**

- Average rating distribution histogram with KDE curve
- Ratings count distribution (log scale)
- Description length distribution (word count)
- Missing values percentage per column
- Sample data saved to `../raw/goodreads_sample.csv`

### Bias Analysis & Fairness Assessment

**Notebook:** [`bias_analysis.ipynb`](datapipeline/data/notebooks/bias_analysis.ipynb)

**Purpose:** Comprehensive bias detection and mitigation analysis using group-level shrinkage to ensure fair recommendations across multiple dimensions.

**Analysis Dimensions:**

1. **Popularity Bias** - High/Medium/Low popularity groups (based on book_popularity_normalized)
2. **Book Length Bias** - Categories based on book_length_category
3. **Book Era Bias** - Publication era groups (book_era)
4. **User Activity Bias** - High/Medium/Low activity user groups (based on user_activity_count)
5. **Reading Pace Bias** - Fast/Medium/Slow reader categories (reading_pace_category)
6. **Author Gender Bias** - Male/Female author groups (using gender_guesser library)

**Technical Implementation:**

- **Data Source:** BigQuery table `goodreads_features_cleaned` with 10,000+ records
- **Mitigation Method:** Group-level shrinkage with λ = 0.5 to pull extreme group means toward global mean
- **Metrics Analyzed:**
  - Average rating per group (before/after mitigation)
  - Percentage read per group (behavioral metric, unchanged)
- **Fairness Calculation:** Equity index based on variance reduction and mean change penalties

**Analysis Process:**

- **Before Analysis:** Calculated group means for each dimension
- **Mitigation Applied:** Adjusted ratings using formula: `rating_debiased = original_rating - λ * (group_mean - global_mean)`
- **After Analysis:** Recalculated group means post-mitigation
- **Fairness Metrics:** Computed variance ratios, relative mean changes, and equity indices

**Key Findings:**

- **Bias Detection:** Systematic rating disparities identified across all analyzed dimensions
- **Mitigation Effectiveness:** Group mean shrinkage successfully reduced extreme group differences
- **Behavioral Preservation:** Reading behavior (% read) remained unchanged to preserve genuine user engagement
- **Fairness Improvement:** Equity index improvements achieved across all dimensions

**Visualizations Generated:**

- Side-by-side bar charts comparing before/after ratings for each dimension
- Side-by-side bar charts showing % read behavior (unchanged)
- Equity index summary bar chart across all dimensions
- Detailed explanations for each dimension's bias patterns and mitigation effects

**Output Files:**

- `fairness_summary_lambda_0_5.csv` - Comprehensive fairness metrics across all dimensions
- Multiple visualization charts showing bias reduction effectiveness

### Project Visuals

**Directory:** [`assets/`](assets/)

<div>
  <p align="center">
    <img src="assets/db_tables_gcp.png" alt="DB Tables on GCP" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>DB Tables on GCP</em>
  </p>

  <p align="center">
    <img src="assets/DAG_task_instances.jpeg" alt="DAG Task Instances" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>DAG Task Instances</em>
  </p>

  <p align="center">
    <img src="assets/DAG_task.jpg" alt="DAG Task" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>DAG Task Details</em>
  </p>

  <p align="center">
    <img src="assets/email_alerts.jpg" alt="Email Alerts" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Email Alerts (Notifications)</em>
  </p>

  <p align="center">
    <img src="assets/gantt_chart.jpeg" alt="Gantt Chart" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>Gantt Chart (Pipeline Schedule)</em>
  </p>

  <p align="center">
    <img src="assets/DVC.png" alt="DVC" style="max-width:600px; width:100%; height:auto; border-radius:8px;" />
    <br/>
    <em>DVC</em>
  </p>
</div>
