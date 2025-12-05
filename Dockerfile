FROM apache/airflow:2.10.3-python3.11

USER root

# Install system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Git configuration arguments
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

# Copy application code with correct ownership
COPY --chown=airflow:root . /opt/airflow/

# Switch to airflow user for everything else
USER airflow

# Configure Git
RUN git config --global user.name "$GIT_USER_NAME" && \
    git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global --add safe.directory /opt/airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Install Airflow providers
RUN pip install --no-cache-dir \
    "apache-airflow-providers-celery>=3.3.0" \
    "apache-airflow-providers-redis" \
    "apache-airflow-providers-postgres"

# Configure DVC if .dvc directory exists
RUN if [ -d /opt/airflow/.dvc ]; then \
        cd /opt/airflow && \
        dvc config core.no_scm false; \
    fi

ENV AIRFLOW_HOME=/opt/airflow