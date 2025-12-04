resource "google_vertex_ai_endpoint" "recommendation_endpoint" {
  provider     = google-beta
  name         = var.endpoint_display_name
  display_name = var.endpoint_display_name
  description  = "Endpoint for Goodreads book recommendation model"
  location     = var.region
  labels       = local.labels

  network = null

  depends_on = [
    google_project_service.required_apis["aiplatform.googleapis.com"]
  ]
}

resource "google_logging_metric" "prediction_latency" {
  name        = "goodreads-prediction-latency"
  description = "Latency of prediction requests"
  filter      = <<-EOT
    resource.type="aiplatform.googleapis.com/Endpoint"
    resource.labels.endpoint_id="${google_vertex_ai_endpoint.recommendation_endpoint.name}"
    severity>=DEFAULT
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "model_version"
      value_type  = "STRING"
      description = "The model version used for prediction"
    }
  }

  bucket_options {
    explicit_buckets {
      bounds = [0, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    }
  }
}

resource "google_logging_metric" "prediction_errors" {
  name        = "goodreads-prediction-errors"
  description = "Count of prediction errors"
  filter      = <<-EOT
    resource.type="aiplatform.googleapis.com/Endpoint"
    resource.labels.endpoint_id="${google_vertex_ai_endpoint.recommendation_endpoint.name}"
    severity>=ERROR
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "deployment_events" {
  name        = "goodreads-deployment-events"
  description = "Model deployment events"
  filter      = <<-EOT
    resource.type="aiplatform.googleapis.com/Endpoint"
    resource.labels.endpoint_id="${google_vertex_ai_endpoint.recommendation_endpoint.name}"
    jsonPayload.event_type=~"DEPLOY|UNDEPLOY"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "event_type"
      value_type  = "STRING"
      description = "Type of deployment event (DEPLOY/UNDEPLOY)"
    }
  }
}

resource "google_monitoring_alert_policy" "prediction_error_rate" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Goodreads Model - High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Prediction Error Rate > 5%"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.prediction_errors.name}\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = "The prediction error rate for the Goodreads recommendation model has exceeded 5%. Please investigate the model endpoint and recent deployments."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "prediction_latency" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Goodreads Model - High Latency"
  combiner     = "OR"

  conditions {
    display_name = "P95 Latency > 2s"

    condition_threshold {
      filter          = "metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 2000

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "604800s"
  }

  documentation {
    content   = "The P95 prediction latency for the Goodreads recommendation model has exceeded 2 seconds. Consider scaling up resources or investigating model performance."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_dashboard" "model_dashboard" {
  dashboard_json = jsonencode({
    displayName = "Goodreads Model Deployment Dashboard"
    gridLayout = {
      columns = 2
      widgets = [
        {
          title = "Prediction Request Count"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"aiplatform.googleapis.com/prediction/online/prediction_count\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        },
        {
          title = "Prediction Latency (P50, P95, P99)"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_50"
                    }
                  }
                }
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_95"
                    }
                  }
                }
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\" AND resource.type=\"aiplatform.googleapis.com/Endpoint\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_99"
                    }
                  }
                }
              }
            ]
          }
        },
        {
          title = "Error Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.prediction_errors.name}\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        },
        {
          title = "Deployment Events"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.deployment_events.name}\""
                  aggregation = {
                    alignmentPeriod    = "3600s"
                    perSeriesAligner   = "ALIGN_SUM"
                    crossSeriesReducer = "REDUCE_SUM"
                    groupByFields      = ["metric.label.event_type"]
                  }
                }
              }
            }]
          }
        }
      ]
    }
  })
}
