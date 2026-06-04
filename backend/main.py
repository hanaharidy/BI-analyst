"""
FastAPI Application
────────────────────
Main entry point for the BI Analyst backend.
Uses the dynamic Supervisor‑based orchestrator (tool‑calling agent).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import structlog

from backend.db.connection import init_db, create_tables
# Replace the old orchestrator with the new supervisor version
from backend.orchestrator_supervisor import BISupervisorOrchestrator
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

# Build LLM config – kept for consistency, but the new orchestrator
# uses its own hardcoded model for now (llama3.1). You can extend it later.
llm_config = {
    "use_openai": settings.use_openai,
    "model": settings.openai_model if settings.use_openai else settings.ollama_model,
    "base_url": settings.ollama_base_url,
    "openai_api_key": settings.openai_api_key,
}


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BI Analyst API (Supervisor‑based dynamic agent)")
    init_db(settings.database_url)
    create_tables()
    # Instantiate the new supervisor orchestrator (no llm_config needed, it uses its own)
    app.state.orchestrator = BISupervisorOrchestrator()
    logger.info("Ready ✓")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AI Business Intelligence Analyst",
    description="Multi‑agent BI system with dynamic tool‑calling supervisor.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models (unchanged) ─────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str


from pydantic import BaseModel, Field

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
    kpis: list[dict] = Field(default_factory=list)   # ✅ forces list even if None
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = app.state.orchestrator.run(req.question)

    # Remove internal fields that are not part of the response model
    result.pop("_pdf_bytes", None)
    result.pop("supervisor_answer", None)   # extra field from supervisor

    return AnalyzeResponse(**{k: v for k, v in result.items() if k in AnalyzeResponse.model_fields})


@app.post("/report/pdf")
async def export_pdf(req: AnalyzeRequest):
    """Rerun pipeline and return PDF report as binary download."""
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
    from backend.db.connection import get_schema_description
    return {"schema": get_schema_description()}