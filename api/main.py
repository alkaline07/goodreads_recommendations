from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from .data_models import (
    LoginRequest,
    LoginResponse,
    BookRecommendation,
    ClickRequest,
    BookDetails,
    MarkReadRequest
)
from .queries import (
    check_user_exists,
    get_top_recommendations,
    get_global_top_recommendations,
    log_ctr_event,
    get_books_read_by_user,
    get_books_not_read_by_user,
    insert_read_interaction
)
from .monitoring_dashboard import router as monitoring_router
from .middleware import MonitoringMiddleware, get_metrics_collector
from fastapi import HTTPException

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup/shutdown.
    Automatically initializes monitoring tables and imports MLflow data if not already set up.
    """
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    
    # Step 1: Initialize monitoring tables
    try:
        from scripts.init_monitoring import ensure_monitoring_setup
        
        logger.info("Checking monitoring setup on startup...")
        success = ensure_monitoring_setup(silent=True)
        
        if success:
            logger.info("Monitoring tables ready")
        else:
            logger.warning("Could not initialize monitoring tables - dashboard may show no data")
    except ImportError as e:
        logger.warning(f"Could not import init_monitoring: {e}")
    except Exception as e:
        logger.warning(f"Error during monitoring setup: {e}")
    
    # Step 2: Import MLflow data if available and not already imported
    try:
        from scripts.import_mlruns_to_monitoring import ensure_mlruns_imported
        
        logger.info("Checking for MLflow data to import...")
        mlruns_path = os.environ.get("MLFLOW_RUNS_PATH", "mlruns")
        success = ensure_mlruns_imported(mlruns_path=mlruns_path, silent=True)
        
        if success:
            logger.info("MLflow data check complete")
        else:
            logger.warning("Could not import MLflow data")
    except ImportError as e:
        logger.debug(f"Could not import mlruns importer: {e}")
    except Exception as e:
        logger.warning(f"Error during MLflow import: {e}")
    
    yield


app = FastAPI(title="Goodreads Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MonitoringMiddleware)

app.include_router(monitoring_router)


@app.get("/metrics")
def get_api_metrics():
    """Get real-time API performance metrics."""
    collector = get_metrics_collector()
    return collector.get_all_stats()


@app.get("/metrics/timeline")
def get_api_timeline(minutes: int = 5):
    """Get API request timeline for charts."""
    collector = get_metrics_collector()
    return collector.get_timeline_data(minutes=minutes)


frontend_metrics_store = []


def get_frontend_metrics_store():
    """Get the frontend metrics store for monitoring dashboard."""
    return frontend_metrics_store


@app.post("/frontend-metrics")
async def receive_frontend_metrics(request: Request):
    """Receive frontend performance metrics from browser clients."""
    try:
        body = await request.body()
        if body:
            import json
            data = json.loads(body)
            data['received_at'] = datetime.utcnow().isoformat()
            frontend_metrics_store.append(data)
            if len(frontend_metrics_store) > 1000:
                frontend_metrics_store.pop(0)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/metrics/frontend")
def get_frontend_metrics():
    """Get aggregated frontend performance metrics."""
    if not frontend_metrics_store:
        return {"sessions": 0, "metrics": {}}
    
    all_api_calls = []
    all_errors = []
    web_vitals = {"lcp": [], "inp": [], "cls": [], "fcp": [], "ttfb": []}
    
    for entry in frontend_metrics_store[-100:]:
        metrics = entry.get("metrics", {})
        all_api_calls.extend(metrics.get("apiCalls", []))
        all_errors.extend(metrics.get("errors", []))
        
        vitals = metrics.get("webVitals", {})
        if vitals.get("lcp"):
            web_vitals["lcp"].append(vitals["lcp"])
        if vitals.get("inp"):
            web_vitals["inp"].append(vitals["inp"])
        if vitals.get("cls") is not None:
            web_vitals["cls"].append(vitals["cls"])
        if vitals.get("fcp"):
            web_vitals["fcp"].append(vitals["fcp"])
        if vitals.get("ttfb"):
            web_vitals["ttfb"].append(vitals["ttfb"])
    
    latencies = [c["latency"] for c in all_api_calls if "latency" in c]
    
    return {
        "sessions": len(set(e.get("sessionId") for e in frontend_metrics_store)),
        "total_api_calls": len(all_api_calls),
        "total_errors": len(all_errors),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "web_vitals": {
            "lcp_avg": sum(web_vitals["lcp"]) / len(web_vitals["lcp"]) if web_vitals["lcp"] else None,
            "inp_avg": sum(web_vitals["inp"]) / len(web_vitals["inp"]) if web_vitals["inp"] else None,
            "cls_avg": sum(web_vitals["cls"]) / len(web_vitals["cls"]) if web_vitals["cls"] else None,
            "fcp_avg": sum(web_vitals["fcp"]) / len(web_vitals["fcp"]) if web_vitals["fcp"] else None,
            "ttfb_avg": sum(web_vitals["ttfb"]) / len(web_vitals["ttfb"]) if web_vitals["ttfb"] else None,
        },
        "recent_errors": all_errors[-10:],
        "timestamp": datetime.utcnow().isoformat()
    }


# ------------------------------------------------------
# SERVE FRONTEND
# ------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/app")
def serve_frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/app/{filename}")
def serve_frontend_file(filename: str):
    """Serve frontend static files."""
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})


# ------------------------------------------------------
# HEALTH CHECK (no external dependencies)
# ------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ------------------------------------------------------
# 1. LOGIN
# ------------------------------------------------------
@app.post("/load-recommendation", response_model=LoginResponse)
def login(request: LoginRequest):
    user_id = request.user_id

    # CASE 1: EXISTING USER → personalized recommendations
    if check_user_exists(user_id):
        df = get_top_recommendations(user_id)

        recommendations = [
            BookRecommendation(
                book_id=row["book_id"],
                title=row["title"],
                author=row["author"],
                predicted_rating=row["predicted_rating"]
            )
            for row in df
        ]

        return LoginResponse(user_id=user_id, recommendations=recommendations)

    # CASE 2: NEW USER → global top 10
    df = get_global_top_recommendations()

    recommendations = [
        BookRecommendation(
            book_id=row["book_id"],
            title=row["title"],
            author=row["author"],
            predicted_rating=row["predicted_rating"]
        )
        for row in df
    ]

    return LoginResponse(user_id=user_id, recommendations=recommendations)


# ------------------------------------------------------
# 2. BOOK CLICK - CTR EVENT
# ------------------------------------------------------
@app.post("/book-click")
def book_click(request: ClickRequest):

    success = log_ctr_event(
        user_id=request.user_id,
        book_id=request.book_id,
        event_type=request.event_type,
        book_title=request.book_title
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to log CTR event"
        )

    return {"success": True}


# ------------------------------------------------------
# 3. USER HAS READ (is_read = TRUE)
# ------------------------------------------------------
@app.get("/books-read/{user_id}")
def books_read(user_id: str):
    rows = get_books_read_by_user(user_id)
    return {"user_id": user_id, "books_read": rows}


# ------------------------------------------------------
# 4. USER HAS NOT READ
# ------------------------------------------------------
@app.get("/books-unread/{user_id}")
def books_unread(user_id: str):
    rows = get_books_not_read_by_user(user_id)
    return {"user_id": user_id, "books_unread": rows}

# ------------------------------------------------------
# 5. USER HAS MARKED AS READ
# ------------------------------------------------------
@app.post("/mark-read")
def mark_read(request: MarkReadRequest):

    success = insert_read_interaction(
        user_id=request.user_id,
        book_id=request.book_id,
        rating=request.rating
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to insert read interaction into BigQuery"
        )

    return {"success": True}