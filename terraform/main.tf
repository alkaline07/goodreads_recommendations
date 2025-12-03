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
