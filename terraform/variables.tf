variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "model_display_name" {
  description = "Display name of the model in Vertex AI Registry"
  type        = string
  default     = "goodreads_boosted_tree_regressor"
}

variable "endpoint_display_name" {
  description = "Display name for the Vertex AI Endpoint"
  type        = string
  default     = "goodreads-recommendation-endpoint"
}

variable "min_replica_count" {
  description = "Minimum number of replicas for auto-scaling"
  type        = number
  default     = 1
}

variable "max_replica_count" {
  description = "Maximum number of replicas for auto-scaling"
  type        = number
  default     = 3
}

variable "machine_type" {
  description = "Machine type for serving predictions"
  type        = string
  default     = "n1-standard-2"
}

variable "accelerator_type" {
  description = "Type of accelerator (GPU) to use, or empty for CPU only"
  type        = string
  default     = ""
}

variable "accelerator_count" {
  description = "Number of accelerators (GPUs) to attach"
  type        = number
  default     = 0
}

variable "enable_logging" {
  description = "Enable prediction request/response logging"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable model monitoring for data drift"
  type        = bool
  default     = true
}

variable "create_service_accounts" {
  description = "Create dedicated service accounts (requires iam.serviceAccounts.create permission)"
  type        = bool
  default     = true
}

variable "create_logging_metrics" {
  description = "Create logging metrics (requires logging.logMetrics.create permission)"
  type        = bool
  default     = true
}

variable "traffic_split_percentage" {
  description = "Percentage of traffic to route to the deployed model (0-100)"
  type        = number
  default     = 100
  
  validation {
    condition     = var.traffic_split_percentage >= 0 && var.traffic_split_percentage <= 100
    error_message = "Traffic split percentage must be between 0 and 100."
  }
}

variable "notification_channels" {
  description = "List of notification channel IDs for alerts"
  type        = list(string)
  default     = []
}

# -----------------------------
# Cloud Run Variables
# -----------------------------
variable "service_name" {
  type        = string
  description = "Cloud Run service name"
  default     = "recommendation-service"
}

variable "image" {
  type        = string
  description = "Container image URL (Artifact Registry)"
}