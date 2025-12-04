terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }

  backend "gcs" {
    bucket = "tf-state-recommendation-system-475301"
    prefix = "cloud-run"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# -----------------------------
# Artifact Registry for Docker images
# -----------------------------
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "recommendation-service"
  description   = "Docker repository for recommendation service images"
  format        = "DOCKER"

  lifecycle {
    ignore_changes = [labels]
  }
}

# -----------------------------
# Cloud Run runtime service account
# -----------------------------
resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-reco-sa"
  display_name = "Cloud Run Recommendation Service SA"
}

resource "google_project_iam_member" "cloud_run_sa_bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "cloud_run_sa_bigquery_data_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "cloud_run_sa_bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# -----------------------------
# Cloud Run v2 service (FastAPI)
# -----------------------------
resource "google_cloud_run_v2_service" "recommendation_service" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = var.image
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.recommendation_service.location
  service  = google_cloud_run_v2_service.recommendation_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}