"""
Insight Generator Agent
────────────────────────
Uses the LLM to extract actionable business insights from analysis results.
Frames findings as clear, executive-level observations with recommendations.
"""

import json
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior business analyst delivering insights to a C-suite executive.

Given statistical analysis results, generate 4–6 concise, actionable insights.
Each insight should be one paragraph (3–4 sentences max).

FORMAT — respond with a JSON array of insight objects:
[
  {
    "type": "positive|warning|opportunity|risk",
    "title": "Short headline (max 8 words)",
    "body": "Detailed insight with specific numbers from the data.",
    "action": "Recommended next step (one sentence)."
  }
]

Rules:
- Always cite specific numbers and percentages from the analysis.
- Be direct — executives have no time for vague language.
- Identify the single most important risk and the single biggest opportunity.
- Do NOT use marketing language. Be analytical and objective.
- Return ONLY the JSON array, no markdown, no preamble.
"""


class InsightGeneratorAgent:
    """Extracts business insights from statistical analysis using LLM reasoning."""

    def __init__(self, llm_config: dict):
        self.llm = self._build_llm(llm_config)
        logger.info("InsightGeneratorAgent ready")

    def _build_llm(self, cfg: dict):
        if cfg.get("use_openai"):
            return ChatOpenAI(
                model=cfg.get("model", "gpt-4o"),
                api_key=cfg.get("openai_api_key"),
                temperature=0.3,  # slight creativity for narrative
            )
        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0.3,
        )

    def generate(self, analysis: dict, intent: dict, question: str) -> list[dict]:
        """
        Returns a list of insight dicts, each with type/title/body/action.
        """
        logger.info("Generating insights", analysis_type=analysis.get("analysis_type"))

        # Trim analysis to avoid overwhelming the context window
        compact_analysis = self._compact_analysis(analysis)

        user_prompt = f"""
Original question: "{question}"

Analysis results:
{json.dumps(compact_analysis, indent=2, default=str)}

Generate business insights.
"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            insights = self._parse_insights(raw)
            logger.info("Insights generated", count=len(insights))
            return insights
        except Exception as exc:
            logger.error("Insight generation failed", error=str(exc))
            return self._fallback_insights(analysis)

    def _compact_analysis(self, analysis: dict) -> dict:
        """Keep only the most relevant parts to stay within token limits."""
        keys_to_keep = [
            "analysis_type", "primary_metric", "summary_stats",
            "rankings", "avg_growth_rate", "total_growth",
            "forecast_next_3", "anomaly_count", "dimension_used",
            "comparison_table",
        ]
        return {k: v for k, v in analysis.items() if k in keys_to_keep}

    def _parse_insights(self, text: str) -> list[dict]:
        import re, json
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        return [data]

    def _fallback_insights(self, analysis: dict) -> list[dict]:
        stats = analysis.get("summary_stats", {})
        return [
            {
                "type": "positive",
                "title": "Analysis completed successfully",
                "body": f"Total revenue: ${stats.get('total', 0):,.0f}. Average: ${stats.get('mean', 0):,.0f}.",
                "action": "Review the charts for detailed breakdown.",
            }
        ]