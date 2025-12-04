# Service accounts - only created if var.create_service_accounts is true
# Requires: iam.serviceAccounts.create permission

resource "google_service_account" "model_deployer" {
  count        = var.create_service_accounts ? 1 : 0
  account_id   = "goodreads-model-deployer"
  display_name = "Goodreads Model Deployer Service Account"
  description  = "Service account for automated model deployment"
}

resource "google_service_account" "model_serving" {
  count        = var.create_service_accounts ? 1 : 0
  account_id   = "goodreads-model-serving"
  display_name = "Goodreads Model Serving Service Account"
  description  = "Service account for model serving endpoints"
}

resource "google_project_iam_member" "deployer_vertex_admin" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.model_deployer[0].email}"
}

resource "google_project_iam_member" "deployer_bigquery_reader" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.model_deployer[0].email}"
}

resource "google_project_iam_member" "deployer_storage_admin" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.model_deployer[0].email}"
}

resource "google_project_iam_member" "deployer_logging" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.model_deployer[0].email}"
}

resource "google_project_iam_member" "serving_vertex_user" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.model_serving[0].email}"
}

resource "google_project_iam_member" "serving_bigquery_reader" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.model_serving[0].email}"
}

resource "google_project_iam_member" "serving_logging" {
  count   = var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.model_serving[0].email}"
}

resource "google_service_account_iam_member" "deployer_can_impersonate_serving" {
  count              = var.create_service_accounts ? 1 : 0
  service_account_id = google_service_account.model_serving[0].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.model_deployer[0].email}"
}