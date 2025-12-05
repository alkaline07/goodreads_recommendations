# Vertex AI Endpoint - this is the main resource
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

# =============================================================================
# ML MODEL MONITORING - Custom Metrics and Alerts
# =============================================================================

# Custom metric descriptors for ML model performance tracking
resource "google_monitoring_metric_descriptor" "ml_rmse" {
  count        = var.enable_monitoring ? 1 : 0
  description  = "Root Mean Square Error for ML model predictions"
  display_name = "Model RMSE"
  type         = "custom.googleapis.com/ml/goodreads/rmse"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"

  labels {
    key         = "model_name"
    value_type  = "STRING"
    description = "Name of the ML model"
  }
}

resource "google_monitoring_metric_descriptor" "ml_mae" {
  count        = var.enable_monitoring ? 1 : 0
  description  = "Mean Absolute Error for ML model predictions"
  display_name = "Model MAE"
  type         = "custom.googleapis.com/ml/goodreads/mae"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"

  labels {
    key         = "model_name"
    value_type  = "STRING"
    description = "Name of the ML model"
  }
}

resource "google_monitoring_metric_descriptor" "ml_r_squared" {
  count        = var.enable_monitoring ? 1 : 0
  description  = "R-squared coefficient for ML model predictions"
  display_name = "Model R-squared"
  type         = "custom.googleapis.com/ml/goodreads/r_squared"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"

  labels {
    key         = "model_name"
    value_type  = "STRING"
    description = "Name of the ML model"
  }
}

resource "google_monitoring_metric_descriptor" "ml_accuracy" {
  count        = var.enable_monitoring ? 1 : 0
  description  = "Prediction accuracy within tolerance"
  display_name = "Model Accuracy"
  type         = "custom.googleapis.com/ml/goodreads/accuracy"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "%"

  labels {
    key         = "model_name"
    value_type  = "STRING"
    description = "Name of the ML model"
  }

  labels {
    key         = "tolerance"
    value_type  = "STRING"
    description = "Tolerance level (0.5 or 1.0)"
  }
}

resource "google_monitoring_metric_descriptor" "data_drift_psi" {
  count        = var.enable_monitoring ? 1 : 0
  description  = "Population Stability Index for data drift detection"
  display_name = "Data Drift PSI"
  type         = "custom.googleapis.com/ml/goodreads/drift_psi"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"

  labels {
    key         = "feature_name"
    value_type  = "STRING"
    description = "Name of the feature being monitored"
  }
}

# Alert policy for model decay detection
resource "google_monitoring_alert_policy" "model_decay_alert" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Goodreads Model Decay Alert"
  combiner     = "OR"

  conditions {
    display_name = "RMSE Exceeds Threshold"

    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/ml/goodreads/rmse\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1.5

      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 3
      }
    }
  }

  conditions {
    display_name = "R-squared Below Threshold"

    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/ml/goodreads/r_squared\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_LT"
      threshold_value = 0.3

      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "86400s"
  }

  documentation {
    content   = <<-EOT
    ## Model Decay Detected

    The Goodreads recommendation model is showing signs of performance degradation.

    **Recommended Actions:**
    1. Check recent data drift reports in `docs/monitoring_reports/`
    2. Review MLflow experiment tracking for metric trends
    3. Consider triggering model retraining if drift is confirmed
    4. Validate data pipeline for any anomalies

    **Dashboard:** Access the monitoring dashboard at `/report`
    EOT
    mime_type = "text/markdown"
  }
}

# Alert policy for data drift detection
resource "google_monitoring_alert_policy" "data_drift_alert" {
  count        = var.enable_monitoring ? 1 : 0
  display_name = "Goodreads Data Drift Alert"
  combiner     = "OR"

  conditions {
    display_name = "PSI Exceeds Threshold"

    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/ml/goodreads/drift_psi\" AND resource.type=\"global\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.2

      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "86400s"
  }

  documentation {
    content   = <<-EOT
    ## Data Drift Detected

    Significant data drift has been detected in one or more features.

    **What this means:**
    - The distribution of input data has shifted from the training baseline
    - Model predictions may become less accurate

    **Recommended Actions:**
    1. Review drift reports in `docs/monitoring_reports/`
    2. Investigate recent data changes
    3. Consider retraining the model with recent data

    **PSI Interpretation:**
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.2: Moderate shift (monitor)
    - PSI >= 0.2: Significant shift (action required)
    EOT
    mime_type = "text/markdown"
  }
}

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