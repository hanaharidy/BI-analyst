"""
Query Understanding Agent
─────────────────────────
Parses user's natural language question into a structured intent object.
Extracts: analysis type, entities, time range, metrics, filters.
"""

import json
import re
from typing import Any
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage
from backend.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a business intelligence query parser. 
Analyze the user's question and extract structured information.

Respond ONLY with a valid JSON object — no markdown, no explanation.

JSON shape:
{
  "analysis_type": "<trend|comparison|ranking|anomaly|forecast|summary>",
  "metrics": ["revenue", "profit", "quantity", ...],
  "dimensions": ["region", "product", "category", "customer", "segment", ...],
  "time_range": {
    "start": "YYYY-MM-DD or null",
    "end": "YYYY-MM-DD or null",
    "period": "Q4 2025 or null",
    "granularity": "day|week|month|quarter|year"
  },
  "filters": {
    "region": null or "North America",
    "category": null or "Electronics",
    "segment": null or "Enterprise",
    "status": "completed"
  },
  "top_n": null or integer,
  "intent_summary": "one sentence plain-English summary of what we need to compute"
}

Examples:
User: "Show me revenue trends by region for Q4 2025"
→ analysis_type: "trend", metrics: ["revenue"], dimensions: ["region"], time_range.period: "Q4 2025"

User: "Which products are underperforming this quarter?"
→ analysis_type: "ranking", metrics: ["revenue"], dimensions: ["product"], time_range.period: "current quarter"
"""


class QueryUnderstandingAgent:
    """Converts free-text questions into structured query intents."""

    def __init__(self, llm_config: dict):
        self.llm = self._build_llm(llm_config)
        logger.info("QueryUnderstandingAgent ready", model=llm_config.get("model"))

    def _build_llm(self, cfg: dict):
        if cfg.get("use_openai"):
                pass

        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0,
        )

    def parse(self, question: str) -> dict[str, Any]:
        """Parse a business question into a structured intent."""
        logger.info("Parsing question", question=question)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}"),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            intent = self._safe_parse_json(raw)
            # Always ensure status filter defaults to completed
            intent.setdefault("filters", {})
            intent["filters"].setdefault("status", "completed")
            logger.info("Intent parsed", analysis_type=intent.get("analysis_type"))
            return intent
        except Exception as exc:
            logger.error("Query parsing failed", error=str(exc))
            return self._fallback_intent(question)

    def _safe_parse_json(self, text: str) -> dict:
        """Strip markdown fences and parse JSON."""
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(cleaned)

    def _fallback_intent(self, question: str) -> dict:
        """Return a safe default intent when LLM fails."""
        return {
            "analysis_type": "summary",
            "metrics": ["revenue"],
            "dimensions": ["region"],
            "time_range": {"start": None, "end": None, "period": None, "granularity": "month"},
            "filters": {"status": "completed"},
            "top_n": None,
            "intent_summary": f"Summarize business data relevant to: {question}",
        }