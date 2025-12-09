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

resource "google_compute_address" "elk_static_ip" {
  count  = var.elk_enabled ? 1 : 0
  name   = "elk-stack-ip"
  region = var.region
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
  count        = var.elk_enabled ? 1 : 0
  name         = "elk-stack"
  machine_type = var.elk_machine_type
  zone         = "${var.region}-a"

  tags = ["elk-stack", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.elk_static_ip[0].address
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    
    # Install Docker
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Configure system for Elasticsearch
    sysctl -w vm.max_map_count=262144
    echo "vm.max_map_count=262144" >> /etc/sysctl.conf
    
    # Create ELK directory
    mkdir -p /opt/elk
    cd /opt/elk
    
    # Create docker-compose.yml for ELK
    cat > docker-compose.yml <<'COMPOSE'
    version: '3.8'
    services:
      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        container_name: elasticsearch
        environment:
          - discovery.type=single-node
          - cluster.name=goodreads-cluster
          - bootstrap.memory_lock=true
          - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
          - xpack.security.enabled=false
        ulimits:
          memlock:
            soft: -1
            hard: -1
        volumes:
          - elasticsearch_data:/usr/share/elasticsearch/data
        ports:
          - "9200:9200"
        healthcheck:
          test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -vq '\"status\":\"red\"'"]
          interval: 30s
          timeout: 10s
          retries: 5
        restart: always

      logstash:
        image: docker.elastic.co/logstash/logstash:8.11.0
        container_name: logstash
        environment:
          - "LS_JAVA_OPTS=-Xms256m -Xmx256m"
        ports:
          - "5044:5044"
          - "9600:9600"
        volumes:
          - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf:ro
        depends_on:
          elasticsearch:
            condition: service_healthy
        restart: always

      kibana:
        image: docker.elastic.co/kibana/kibana:8.11.0
        container_name: kibana
        environment:
          - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
        ports:
          - "5601:5601"
        depends_on:
          elasticsearch:
            condition: service_healthy
        restart: always

    volumes:
      elasticsearch_data:
    COMPOSE
    
    # Create Logstash config
    cat > logstash.conf <<'LOGSTASH'
    input {
      beats { port => 5044 }
      http { port => 8080 codec => json }
    }
    filter {
      if [message] =~ /^\{.*\}$/ {
        json { source => "message" target => "parsed_json" }
      }
    }
    output {
      elasticsearch {
        hosts => ["elasticsearch:9200"]
        index => "goodreads-logs-%%{+YYYY.MM.dd}"
      }
    }
    LOGSTASH
    
    # Start ELK stack
    docker compose up -d
    
    echo "ELK stack started successfully"
  EOF

  service_account {
    email  = var.create_service_accounts ? google_service_account.elk_sa[0].email : null
    scopes = ["cloud-platform"]
  }

  labels = local.labels

  allow_stopping_for_update = true
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
