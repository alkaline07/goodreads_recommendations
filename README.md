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

### 3. Run Training Pipeline

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

## Data Analysis & Insights

This section highlights key findings from our exploratory data analysis and bias detection notebooks that inform our recommendation system design.

### Raw Data Analysis [(`raw_analysis.ipynb`)](https://github.com/purva-agarwal/goodreads_recommendations/blob/master/datapipeline/data/notebooks/raw_analysis.ipynb)

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

### Bias Analysis & Fairness Assessment [(`bias_analysis.ipynb`)](https://github.com/purva-agarwal/goodreads_recommendations/blob/master/datapipeline/data/notebooks/bias_analysis.ipynb)

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

### Project Visuals [(assets/)](https://github.com/purva-agarwal/goodreads_recommendations/blob/master/assets)

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