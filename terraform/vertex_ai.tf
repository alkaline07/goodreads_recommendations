# Vertex AI Endpoint - this is the main resource
resource "google_vertex_ai_endpoint" "recommendation_endpoint" {
  provider     = google-beta
  name         = var.endpoint_display_name
  display_name = var.endpoint_display_name
  description  = "Endpoint for Goodreads book recommendation model"
  location     = "us-central1"
  labels       = local.labels

  network = null

  depends_on = [
    google_project_service.required_apis["aiplatform.googleapis.com"]
  ]
}

# Logging metrics - only created if var.create_logging_metrics is true
# Requires: logging.logMetrics.create permission

resource "google_logging_metric" "prediction_count" {
  count       = var.create_logging_metrics ? 1 : 0
  name        = "goodreads-prediction-count"
  description = "Count of prediction requests"
  filter      = "resource.type=\"aiplatform.googleapis.com/Endpoint\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "prediction_errors" {
  count       = var.create_logging_metrics ? 1 : 0
  name        = "goodreads-prediction-errors"
  description = "Count of prediction errors"
  filter      = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND severity>=ERROR"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "deployment_events" {
  count       = var.create_logging_metrics ? 1 : 0
  name        = "goodreads-deployment-events"
  description = "Model deployment events"
  filter      = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND (jsonPayload.event_type=\"DEPLOY\" OR jsonPayload.event_type=\"UNDEPLOY\")"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Alert policies - disabled until endpoint has traffic
# These require actual prediction data to work properly
# Uncomment after your endpoint is serving predictions

# resource "google_monitoring_alert_policy" "prediction_error_rate" {
#   count        = var.enable_monitoring && var.create_logging_metrics ? 1 : 0
#   display_name = "Goodreads Model - High Error Rate"
#   combiner     = "OR"
#
#   conditions {
#     display_name = "Prediction Error Rate > 5/min"
#
#     condition_threshold {
#       filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.prediction_errors[0].name}\" AND resource.type=\"global\""
#       duration        = "300s"
#       comparison      = "COMPARISON_GT"
#       threshold_value = 5
#
#       aggregations {
#         alignment_period   = "60s"
#         per_series_aligner = "ALIGN_RATE"
#       }
#     }
#   }
#
#   notification_channels = var.notification_channels
#
#   alert_strategy {
#     auto_close = "604800s"
#   }
#
#   documentation {
#     content   = "The prediction error count for the Goodreads recommendation model has exceeded 5 per minute. Please investigate the model endpoint and recent deployments."
#     mime_type = "text/markdown"
#   }
#
#   depends_on = [google_logging_metric.prediction_errors]
# }

# Dashboard - only created if enable_monitoring is true
resource "google_monitoring_dashboard" "model_dashboard" {
  count = var.enable_monitoring ? 1 : 0
  dashboard_json = jsonencode({
    displayName = "Goodreads Model Deployment Dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Prediction Request Count"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/prediction_count\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Prediction Latency (P50, P95, P99)"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_50"
                      }
                    }
                  }
                  plotType = "LINE"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                      }
                    }
                  }
                  plotType = "LINE"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/response_latencies\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_99"
                      }
                    }
                  }
                  plotType = "LINE"
                }
              ]
              yAxis = {
                scale = "LINEAR"
              }
            }
          }
        },
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Replica Count"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/replicas\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "CPU Utilization"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND metric.type=\"aiplatform.googleapis.com/prediction/online/cpu/utilization\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}