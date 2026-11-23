"""FastAPI Application for Earthquake Classifier (ETL & EDA Pipeline).

Serves dynamic EDA visualizations, analytics endpoints, and real-time ETL triggers.
Designed for cloud deployment (Vercel serverless / Docker / ASGI).
"""

from __future__ import annotations
import logging
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.eda_service import EDAService
from etl.run_pipeline import run_etl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI-App")

app = FastAPI(
    title="Earthquake Classifier - ETL & EDA Pipeline",
    description="Automated telemetry ETL, Pydantic validation, Cloud SQL loading, and dynamic EDA analytics.",
    version="1.0.0",
)

# Enable CORS for cloud frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template configuration
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

eda_service = EDAService()


class ETLTriggerRequest(BaseModel):
    feed: Optional[str] = "all_week"
    database_url: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the comprehensive EDA Dashboard with Tailwind CSS."""
    summary = eda_service.get_summary()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"summary": summary}
    )


@app.get("/api/eda/summary")
async def get_summary_metrics():
    """Returns top-level telemetry KPIs and target distribution statistics."""
    return eda_service.get_summary()


@app.get("/api/eda/class-balance")
async def get_class_balance():
    """Returns the class balance ratio between minor tremors (< 5.0) and significant earthquakes (>= 5.0)."""
    return eda_service.get_class_balance()


@app.get("/api/eda/missingness")
async def get_missingness_analysis():
    """Returns the missingness percentage across sensor attributes grouped by earthquake class."""
    return eda_service.get_missingness_analysis()


@app.get("/api/eda/spatial")
async def get_spatial_distribution(limit: int = 2500):
    """Returns global event coordinates and magnitude data for tectonic boundary map clusters."""
    return eda_service.get_spatial_distribution(limit=limit)


@app.get("/api/eda/recent")
async def get_recent_earthquakes(limit: int = 50):
    """Returns recent cleaned seismic telemetry records for data preview."""
    return eda_service.get_recent_earthquakes(limit=limit)


@app.post("/api/etl/trigger")
async def trigger_pipeline(payload: ETLTriggerRequest):
    """Dynamically triggers the USGS ETL pipeline to append fresh seismic records to Cloud SQL."""
    try:
        result = run_etl(feed=payload.feed or "all_week", database_url=payload.database_url)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"ETL Execution Trigger Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Cloud health check endpoint for Vercel and uptime monitoring."""
    return {
        "status": "healthy",
        "service": "Earthquake Classifier ETL & EDA",
        "database": os.getenv("DATABASE_URL", "sqlite:///./earthquakes.db").split(":")[0],
    }
