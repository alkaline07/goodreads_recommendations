terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}


resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-reco-sa"
  display_name = "Cloud Run Recommendation Service SA"
}


resource "google_project_iam_member" "cloud_run_sa_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}


resource "google_cloud_run_v2_service" "recommendation_service" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      # This image is passed from GitHub Actions as TF_VAR_image
      image = var.image

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
terraform {
  backend "gcs" {
    bucket = "tf-state-recommendation-system-475301"
    prefix = "cloud-run"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}
