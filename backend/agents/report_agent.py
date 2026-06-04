"""
Report Writer Agent (LLM‑Enhanced)
───────────────────────────────────
Adds a dynamic executive summary written by LLM before the static sections.
"""

from __future__ import annotations
import io
import base64
import json
from datetime import datetime
from jinja2 import Environment, BaseLoader
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# HTML template same as before (I'll omit it here for brevity, but keep your REPORT_HTML)
REPORT_HTML = """ ... (same as your original) ... """

class ReportWriterAgent:
    def __init__(self, llm_config: dict):
        self.llm = self._build_llm(llm_config)
        logger.info("LLM‑enhanced ReportWriterAgent ready")

    def _build_llm(self, cfg: dict):
        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0,
        )

    def compile(
        self,
        question: str,
        intent: dict,
        analysis: dict,
        insights: list[dict],
        charts: list[dict],
        sql_query: str,
    ) -> dict:
        # Generate dynamic executive summary using LLM
        executive_summary = self._generate_summary(question, analysis, insights)

        kpis = self._build_kpis(analysis)
        markdown = self._build_markdown(question, kpis, insights, sql_query, executive_summary)
        html = self._build_html(question, kpis, insights, charts, sql_query, executive_summary)
        pdf_bytes = self._render_pdf(html)

        return {
            "markdown": markdown,
            "html": html,
            "pdf_bytes": pdf_bytes,
            "kpis": kpis,
        }

    def _generate_summary(self, question: str, analysis: dict, insights: list[dict]) -> str:
        """LLM writes a 2‑3 sentence executive summary."""
        prompt = f"""
Question: "{question}"
Key stats: {analysis.get('summary_stats', {})}
Insights: {json.dumps(insights[:2], indent=2, default=str)}
Write a brief executive summary (2‑3 sentences) that answers the question directly and highlights the most important finding.
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content if hasattr(response, "content") else str(response)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return "Analysis completed. See insights and charts below."

    # ── Builders (updated to include summary) ────────────────────────────────
    def _build_markdown(self, question, kpis, insights, sql, summary):
        lines = [
            f"## 📊 Report: {question}\n",
            f"*Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n",
            "### Executive Summary\n",
            f"{summary}\n",
        ]
        if kpis:
            lines.append("### Key Metrics\n")
            for k in kpis:
                lines.append(f"- **{k['label']}**: {k['value']}")
            lines.append("")
        if insights:
            lines.append("### Insights\n")
            icon_map = {"positive": "✅", "warning": "⚠️", "risk": "🔴", "opportunity": "🟣"}
            for ins in insights:
                icon = icon_map.get(ins.get("type", ""), "💡")
                lines.append(f"**{icon} {ins['title']}**")
                lines.append(f"{ins['body']}")
                lines.append(f"*→ {ins['action']}*\n")
        if sql:
            lines.append("### SQL Query\n```sql")
            lines.append(sql)
            lines.append("```")
        return "\n".join(lines)

    def _build_html(self, question, kpis, insights, charts, sql, summary):
        # Same as before but add summary section in HTML
        # (I'll keep your original HTML but inject summary)
        charts_with_b64 = []
        for chart in charts:
            png = chart.get("figure_png")
            b64 = base64.b64encode(png).decode() if png else None
            charts_with_b64.append({**chart, "figure_png_b64": b64})

        env = Environment(loader=BaseLoader())
        template = env.from_string(REPORT_HTML)
        return template.render(
            question=question,
            generated_at=datetime.now().strftime("%B %d, %Y %H:%M"),
            kpis=kpis,
            insights=insights,
            charts=charts_with_b64,
            sql_query=sql,
            executive_summary=summary,   # add this to HTML template if needed
        )

    # _build_kpis and _render_pdf remain exactly the same as your original
    def _build_kpis(self, analysis: dict) -> list[dict]:
        # ... unchanged ...
        pass

    def _render_pdf(self, html: str) -> bytes | None:
        # ... unchanged ...
        pass