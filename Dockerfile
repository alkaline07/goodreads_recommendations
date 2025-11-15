FROM apache/airflow:2.10.3-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gcc && \
    apt-get clean

ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

USER airflow
# Git identity for commits done by Airflow/DVC inside the container
RUN git config --global user.name "$GIT_USER_NAME" && \
    git config --global user.email "$GIT_USER_EMAIL"

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install "apache-airflow-providers-celery>=3.3.0"
RUN pip install apache-airflow-providers-redis
RUN pip install apache-airflow-providers-postgres

COPY . /opt/airflow/
ENV AIRFLOW_HOME=/opt/airflow
