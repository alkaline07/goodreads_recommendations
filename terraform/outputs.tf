output "endpoint_id" {
  description = "The ID of the Vertex AI Endpoint"
  value       = google_vertex_ai_endpoint.recommendation_endpoint.name
}

output "endpoint_resource_name" {
  description = "The full resource name of the Vertex AI Endpoint"
  value       = google_vertex_ai_endpoint.recommendation_endpoint.id
}

output "endpoint_predict_url" {
  description = "The prediction URL for the endpoint"
  value       = "https://${var.region}-aiplatform.googleapis.com/v1/${google_vertex_ai_endpoint.recommendation_endpoint.id}:predict"
}

output "deployer_service_account" {
  description = "Service account email for model deployment"
  value       = var.create_service_accounts ? google_service_account.model_deployer[0].email : null
}

output "serving_service_account" {
  description = "Service account email for model serving"
  value       = var.create_service_accounts ? google_service_account.model_serving[0].email : null
}


output "monitoring_dashboard_url" {
  description = "URL to the Cloud Monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/builder/${google_monitoring_dashboard.model_dashboard.id}?project=${var.project_id}"
}

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}
