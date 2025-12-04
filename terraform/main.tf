terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0.0"
    }
  }

  backend "gcs" {
    bucket = "recommendation-system-475301-terraform-state"
    prefix = "model-deployment"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

locals {
  timestamp = formatdate("YYYYMMDDhhmmss", timestamp())

  labels = {
    project     = "goodreads-recommendations"
    environment = var.environment
    managed_by  = "terraform"
  }
}

data "google_project" "current" {}


resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "bigquery.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# -----------------------------
# Cloud Run runtime service account
# -----------------------------
resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-reco-sa"
  display_name = "Cloud Run Recommendation Service SA"
}


# -----------------------------
# Cloud Run v2 service (FastAPI)
# -----------------------------
resource "google_cloud_run_v2_service" "recommendation_service" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = var.image   # passed from GitHub Actions via TF_VAR_image

      env {
        name  = "PORT"
        value = "8080"
      }
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
