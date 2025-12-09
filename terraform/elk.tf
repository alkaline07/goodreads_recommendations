# -----------------------------
# ELK Stack Infrastructure
# Persistent Compute Engine VM for log aggregation
# -----------------------------

variable "elk_enabled" {
  description = "Enable ELK stack deployment"
  type        = bool
  default     = true
}

variable "elk_machine_type" {
  description = "Machine type for ELK VM (needs 4GB+ RAM for Elasticsearch)"
  type        = string
  default     = "e2-standard-2"
}

variable "elk_disk_size" {
  description = "Persistent disk size in GB for ELK data"
  type        = number
  default     = 50
}

variable "elk_image" {
  description = "ELK container image URL"
  type        = string
  default     = ""
}

resource "google_compute_address" "elk_static_ip" {
  count  = var.elk_enabled ? 1 : 0
  name   = "elk-stack-ip"
  region = var.region
}

resource "google_compute_disk" "elk_data" {
  count = var.elk_enabled ? 1 : 0
  name  = "elk-data-disk"
  type  = "pd-ssd"
  zone  = "${var.region}-a"
  size  = var.elk_disk_size

  labels = local.labels
}

resource "google_service_account" "elk_sa" {
  count        = var.elk_enabled && var.create_service_accounts ? 1 : 0
  account_id   = "elk-stack-sa"
  display_name = "ELK Stack Service Account"
}

resource "google_project_iam_member" "elk_sa_logging" {
  count   = var.elk_enabled && var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.elk_sa[0].email}"
}

resource "google_project_iam_member" "elk_sa_monitoring" {
  count   = var.elk_enabled && var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.elk_sa[0].email}"
}

resource "google_project_iam_member" "elk_sa_artifact_reader" {
  count   = var.elk_enabled && var.create_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.elk_sa[0].email}"
}

resource "google_compute_instance" "elk_stack" {
  count        = var.elk_enabled && var.elk_image != "" ? 1 : 0
  name         = "elk-stack"
  machine_type = var.elk_machine_type
  zone         = "${var.region}-a"

  tags = ["elk-stack", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 30
      type  = "pd-ssd"
    }
  }

  attached_disk {
    source      = google_compute_disk.elk_data[0].self_link
    device_name = "elk-data"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.elk_static_ip[0].address
    }
  }

  metadata = {
    gce-container-declaration = yamlencode({
      spec = {
        containers = [{
          name  = "elk-stack"
          image = var.elk_image
          env = [
            { name = "ES_JAVA_OPTS", value = "-Xms1g -Xmx1g" },
            { name = "LS_JAVA_OPTS", value = "-Xms512m -Xmx512m" }
          ]
          volumeMounts = [{
            name      = "elk-data"
            mountPath = "/var/lib/elasticsearch"
          }]
        }]
        volumes = [{
          name = "elk-data"
          gcePersistentDisk = {
            pdName = "elk-data"
            fsType = "ext4"
          }
        }]
        restartPolicy = "Always"
      }
    })
    google-logging-enabled = "true"
  }

  service_account {
    email  = var.create_service_accounts ? google_service_account.elk_sa[0].email : null
    scopes = ["cloud-platform"]
  }

  labels = local.labels

  allow_stopping_for_update = true

  lifecycle {
    ignore_changes = [metadata["gce-container-declaration"]]
  }
}

resource "google_compute_firewall" "elk_firewall" {
  count   = var.elk_enabled ? 1 : 0
  name    = "allow-elk-ports"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9200", "5601", "5044"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["elk-stack"]

  description = "Allow access to ELK stack ports (Elasticsearch, Kibana, Logstash)"
}

resource "google_compute_firewall" "elk_internal" {
  count   = var.elk_enabled ? 1 : 0
  name    = "allow-elk-internal"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9200", "9300", "5601", "5044", "9600"]
  }

  source_tags = ["elk-stack"]
  target_tags = ["elk-stack"]

  description = "Allow internal ELK communication"
}
