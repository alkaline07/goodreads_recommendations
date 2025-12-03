resource "google_service_account" "model_deployer" {
  account_id   = "goodreads-model-deployer"
  display_name = "Goodreads Model Deployer Service Account"
  description  = "Service account for automated model deployment"
}

resource "google_service_account" "model_serving" {
  account_id   = "goodreads-model-serving"
  display_name = "Goodreads Model Serving Service Account"
  description  = "Service account for model serving endpoints"
}

resource "google_project_iam_member" "deployer_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.model_deployer.email}"
}

resource "google_project_iam_member" "deployer_bigquery_reader" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.model_deployer.email}"
}

resource "google_project_iam_member" "deployer_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.model_deployer.email}"
}

resource "google_project_iam_member" "deployer_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.model_deployer.email}"
}

resource "google_project_iam_member" "serving_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.model_serving.email}"
}

resource "google_project_iam_member" "serving_bigquery_reader" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.model_serving.email}"
}

resource "google_project_iam_member" "serving_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.model_serving.email}"
}

resource "google_service_account_iam_member" "deployer_can_impersonate_serving" {
  service_account_id = google_service_account.model_serving.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.model_deployer.email}"
}
