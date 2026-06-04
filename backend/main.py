"""
FastAPI Application
────────────────────
Main entry point for the BI Analyst backend.
Exposes REST endpoints consumed by the Streamlit frontend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import structlog

from backend.db.connection import init_db, create_tables
from backend.agents.orchestrator_dynamic import BIOrchestrator
from backend.utils.logger import setup_logging, get_logger


# ── Config ─────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:password@localhost:5432/bi_analyst"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    use_openai: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

# Build LLM config from settings
llm_config = {
    "use_openai": settings.use_openai,
    "model": settings.openai_model if settings.use_openai else settings.ollama_model,
    "base_url": settings.ollama_base_url,
    "openai_api_key": settings.openai_api_key,
}


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BI Analyst API")
    init_db(settings.database_url)
    create_tables()
    app.state.orchestrator = BIOrchestrator(llm_config)
    logger.info("Ready ✓")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AI Business Intelligence Analyst",
    description="Multi-agent BI system — ask questions in plain English.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str


class AnalyzeResponse(BaseModel):
    status: str
    question: str
    intent: dict | None = None
    sql: str | None = None
    data_columns: list[str] = []
    analysis: dict = {}
    charts: list[dict] = []
    insights: list[dict] = []
    report_markdown: str = ""
    kpis: list[dict] = []
    pdf_available: bool = False
    elapsed_seconds: float = 0
    steps: list[dict] = []
    error: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Main endpoint: takes a plain-English business question,
    runs the full agent pipeline, and returns results.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = app.state.orchestrator.run(req.question)

    # Strip internal PDF bytes before serializing (accessed via /report/pdf)
    result.pop("_pdf_bytes", None)
    result.pop("data", None)  # omit raw rows from API response (can be large)

    return AnalyzeResponse(**{k: v for k, v in result.items() if k in AnalyzeResponse.model_fields})


@app.post("/report/pdf")
async def export_pdf(req: AnalyzeRequest):
    """Re-runs pipeline and returns PDF report as binary download."""
    result = app.state.orchestrator.run(req.question)
    pdf = result.get("_pdf_bytes")

    if not pdf:
        raise HTTPException(
            status_code=503,
            detail="PDF export unavailable. Ensure WeasyPrint is installed."
        )

    filename = req.question[:40].replace(" ", "_").lower() + ".pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/schema")
async def get_schema():
    """Returns the database schema for debugging / frontend display."""
    from backend.db.connection import get_schema_description
    return {"schema": get_schema_description()}