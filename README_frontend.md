# Frontend Deployment Guide

This guide covers the Streamlit-based frontend application for the Goodreads Book Recommendation System.

## Overview

The frontend is a Streamlit application that provides:
- **Book Recommendations**: Personalized book suggestions based on user reading history
- **Book Search**: Search functionality to find and explore books
- **User Dashboard**: View read/unread books and manage preferences
- **Monitoring Page**: Quick access to API and model performance metrics

## Architecture

```
┌─────────────────────┐      ┌─────────────────────┐     ┌─────────────────────┐
│   Streamlit App     │────▶│   FastAPI Backend   │────▶│   BigQuery ML       │
│   (Cloud Run)       │      │   (Cloud Run)       │     │   (Predictions)     │
│   Port: 8501        │      │   Port: 8080        │     │                     │
└─────────────────────┘      └─────────────────────┘     └─────────────────────┘
```

## File Structure

```
pages/app/
├── __init__.py
├── book_recommendation_app.py    # Main application entry point
└── monitoring_page.py            # Admin monitoring dashboard
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- GCP credentials (for Cloud Run deployment)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install streamlit requests pandas
   ```

2. **Run the application**:
   ```bash
   streamlit run pages/app/book_recommendation_app.py
   ```

3. **Access the app**: Open http://localhost:8501

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -f Dockerfile.frontend -t recommendation-frontend .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 recommendation-frontend
   ```

3. **Access the app**: Open http://localhost:8501

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | Backend API endpoint | `https://recommendation-service-491512947755.us-central1.run.app` |
| `PORT` | Application port | `8501` |

### Changing the Backend API URL

To point to a different backend, modify `API_BASE_URL` in:
- `pages/app/book_recommendation_app.py` (line 13)
- `pages/app/monitoring_page.py` (line 17)

Or set the environment variable when running:
```bash
docker run -p 8501:8501 -e API_BASE_URL=https://your-api.run.app recommendation-frontend
```

## Cloud Run Deployment

### Manual Deployment

1. **Authenticate with GCP**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Configure Docker for Artifact Registry**:
   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
   ```

3. **Build and push the image**:
   ```bash
   IMAGE="us-central1-docker.pkg.dev/YOUR_PROJECT_ID/recommendation-service/recommendation-frontend:latest"
   docker build -f Dockerfile.frontend -t $IMAGE .
   docker push $IMAGE
   ```

4. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy recommendation-frontend \
     --image $IMAGE \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --port 8501
   ```

### Automated Deployment (GitHub Actions)

The frontend automatically deploys when changes are pushed to `master` in:
- `pages/app/**`
- `Dockerfile.frontend`

See `.github/workflows/deploy-frontend.yml` for the workflow configuration.

**Required GitHub Secrets**:
| Secret | Description |
|--------|-------------|
| `GCP_CREDENTIALS` | GCP service account JSON key |
| `GCP_PROJECT_ID` | GCP project ID (default: `recommendation-system-475301`) |
| `GCP_REGION` | GCP region (default: `us-central1`) |

## Features

### Book Recommendations

The main page allows users to:
1. Enter a user ID to get personalized recommendations
2. View book details including cover images, ratings, and descriptions
3. Track click events for recommendation analytics

### Book Search

Users can search the book database by:
- Title
- Author name

Results are fetched from Google Books API for rich metadata.

### Monitoring Dashboard

Access via the Streamlit sidebar to view:
- API performance metrics (latency, error rates, throughput)
- Model performance indicators
- Data drift detection status

### Web Vitals Tracking

The app automatically tracks Core Web Vitals:
- **LCP** (Largest Contentful Paint)
- **FCP** (First Contentful Paint)
- **CLS** (Cumulative Layout Shift)
- **INP** (Interaction to Next Paint)
- **TTFB** (Time to First Byte)

Metrics are sent to the backend `/frontend-metrics` endpoint.

## API Endpoints Used

The frontend interacts with these backend endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load-recommendation` | POST | Get personalized recommendations |
| `/books-read/{user_id}` | GET | Get user's read books |
| `/books-unread/{user_id}` | GET | Get user's unread books |
| `/log-click-event` | POST | Track recommendation clicks |
| `/frontend-metrics` | POST | Submit Web Vitals data |
| `/metrics` | GET | Fetch API performance metrics |

## Troubleshooting

### App won't start

**Symptom**: Container exits immediately or Streamlit doesn't load.

**Solutions**:
1. Check if port 8501 is available:
   ```bash
   lsof -i :8501
   ```
2. Verify the Dockerfile entry point:
   ```bash
   docker run -it recommendation-frontend /bin/bash
   streamlit run pages/app/book_recommendation_app.py --server.port=8501
   ```

### Books database warning

**Symptom**: "books_database.json not found, using fallback data"

**Cause**: This is expected. The app uses a fallback database with sample books when the JSON file isn't present.

**Solution**: Place `books_database.json` in the `data/` directory if you have the full dataset.

### API connection errors

**Symptom**: "Error fetching recommendations" or API timeout errors.

**Solutions**:
1. Verify the backend is running:
   ```bash
   curl https://recommendation-service-491512947755.us-central1.run.app/health
   ```
2. Check CORS configuration on the backend
3. Verify network connectivity from Cloud Run

### Monitoring page errors

**Symptom**: Monitoring page fails to load metrics.

**Solutions**:
1. Ensure `pandas` is installed
2. Verify the `/metrics` endpoint is accessible
3. Check the API_BASE_URL configuration

## Development

### Adding New Pages

1. Create a new `.py` file in `pages/app/`
2. Use Streamlit's page config:
   ```python
   import streamlit as st
   st.set_page_config(page_title="My Page", page_icon="📚")
   ```
3. Rebuild and redeploy the Docker image

### Testing Locally with Mock Data

The app includes fallback data for testing without the full database:
```python
def load_fallback_database():
    return [
        {"book_id": "13079104", "title": "Circe", "author": "Madeline Miller", ...},
        {"book_id": "11295686", "title": "Gone Girl", "author": "Gillian Flynn", ...},
    ]
```

## Related Documentation

- [Project Replication Guide](README_project_replication.md) - Full system setup
- [API Backend](api/README.md) - Backend API documentation
- [Model Training](README_model.md) - ML pipeline documentation
