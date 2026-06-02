"""
Report Writer Agent
────────────────────
Compiles all agent outputs into a structured executive report.
Renders as Markdown (for Streamlit display) and PDF (for download).
"""

from __future__ import annotations
import io
import base64
from datetime import datetime
from jinja2 import Environment, BaseLoader
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ── HTML/CSS template for PDF ─────────────────────────────────────────────────

REPORT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; color: #1F2937; background: #fff; font-size: 13px; }

  .header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    color: white; padding: 36px 48px;
  }
  .header h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
  .header .meta { font-size: 12px; opacity: 0.75; }

  .body { padding: 32px 48px; }

  .section-title {
    font-size: 16px; font-weight: 700; color: #1E3A5F;
    border-bottom: 2px solid #DBEAFE; padding-bottom: 6px; margin: 24px 0 14px;
  }

  .kpi-row { display: flex; gap: 16px; margin-bottom: 20px; }
  .kpi-card {
    flex: 1; background: #F8FAFC; border-left: 4px solid #2563EB;
    padding: 14px 18px; border-radius: 6px;
  }
  .kpi-card .label { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-card .value { font-size: 22px; font-weight: 700; color: #1E3A5F; margin-top: 2px; }

  .insight {
    padding: 12px 16px; margin-bottom: 10px; border-radius: 6px;
    border-left: 4px solid #ccc; background: #F9FAFB;
  }
  .insight.positive { border-color: #10B981; }
  .insight.warning  { border-color: #F59E0B; }
  .insight.risk     { border-color: #EF4444; }
  .insight.opportunity { border-color: #6366F1; }

  .insight h4 { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .insight p  { font-size: 12px; color: #374151; line-height: 1.5; }
  .insight .action { font-size: 11px; color: #6B7280; margin-top: 6px; font-style: italic; }

  .chart-section { margin: 20px 0; }
  .chart-section img { width: 100%; border-radius: 6px; border: 1px solid #E5E7EB; }

  .sql-block {
    background: #1E293B; color: #E2E8F0; padding: 14px; border-radius: 6px;
    font-family: 'Courier New', monospace; font-size: 11px; line-height: 1.6;
    white-space: pre-wrap; word-break: break-all; margin-top: 8px;
  }

  .footer {
    margin-top: 32px; padding-top: 16px; border-top: 1px solid #E5E7EB;
    font-size: 10px; color: #9CA3AF; text-align: center;
  }
</style>
</head>
<body>

<div class="header">
  <h1>📊 Business Intelligence Report</h1>
  <div class="meta">
    Generated: {{ generated_at }} &nbsp;|&nbsp; Question: "{{ question }}"
  </div>
</div>

<div class="body">

  {% if kpis %}
  <div class="section-title">Key Metrics</div>
  <div class="kpi-row">
    {% for kpi in kpis %}
    <div class="kpi-card">
      <div class="label">{{ kpi.label }}</div>
      <div class="value">{{ kpi.value }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if insights %}
  <div class="section-title">Executive Insights</div>
  {% for insight in insights %}
  <div class="insight {{ insight.type }}">
    <h4>{{ insight.title }}</h4>
    <p>{{ insight.body }}</p>
    <p class="action">→ {{ insight.action }}</p>
  </div>
  {% endfor %}
  {% endif %}

  {% if charts %}
  <div class="section-title">Visualizations</div>
  {% for chart in charts %}
  <div class="chart-section">
    <strong>{{ chart.title }}</strong>
    {% if chart.figure_png_b64 %}
    <img src="data:image/png;base64,{{ chart.figure_png_b64 }}" alt="{{ chart.title }}">
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if sql_query %}
  <div class="section-title">Query Used</div>
  <div class="sql-block">{{ sql_query }}</div>
  {% endif %}

</div>

<div class="footer">
  AI-Powered BI Analyst &nbsp;|&nbsp; Confidential &nbsp;|&nbsp; {{ generated_at }}
</div>

</body>
</html>
"""


class ReportWriterAgent:
    """Compiles all outputs into a final report (Markdown + HTML/PDF)."""

    def compile(
        self,
        question: str,
        intent: dict,
        analysis: dict,
        insights: list[dict],
        charts: list[dict],
        sql_query: str,
    ) -> dict:
        """
        Returns:
          - markdown: str       → for Streamlit display
          - html: str           → rendered HTML
          - pdf_bytes: bytes    → WeasyPrint PDF
          - kpis: list[dict]    → key metric cards
        """
        logger.info("Compiling report", insights=len(insights), charts=len(charts))

        kpis = self._build_kpis(analysis)
        markdown = self._build_markdown(question, kpis, insights, sql_query)
        html = self._build_html(question, kpis, insights, charts, sql_query)
        pdf_bytes = self._render_pdf(html)

        return {
            "markdown": markdown,
            "html": html,
            "pdf_bytes": pdf_bytes,
            "kpis": kpis,
        }

    # ── Builders ──────────────────────────────────────────────────────────────

    def _build_kpis(self, analysis: dict) -> list[dict]:
        kpis = []
        stats = analysis.get("summary_stats", {})

        if stats.get("total") is not None:
            metric = analysis.get("primary_metric", "revenue")
            label = metric.replace("_", " ").title()
            kpis.append({"label": f"Total {label}", "value": f"${stats['total']:,.0f}"})

        if stats.get("mean") is not None:
            kpis.append({"label": "Average", "value": f"${stats['mean']:,.0f}"})

        growth = analysis.get("avg_growth_rate")
        if growth is not None:
            kpis.append({"label": "Avg Growth", "value": f"{growth:+.1f}%"})

        forecast = analysis.get("forecast_next_3", [])
        if forecast:
            kpis.append({"label": "Next Period (Forecast)", "value": f"${forecast[0]:,.0f}"})

        return kpis[:4]  # max 4 KPI cards

    def _build_markdown(self, question: str, kpis: list, insights: list, sql: str) -> str:
        lines = [
            f"## 📊 Report: {question}\n",
            f"*Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n",
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

    def _build_html(
        self, question: str, kpis: list, insights: list, charts: list, sql: str
    ) -> str:
        # Encode chart PNGs as base64 for embedding
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
        )

    def _render_pdf(self, html: str) -> bytes | None:
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html).write_pdf()
            logger.info("PDF rendered", size_kb=round(len(pdf_bytes) / 1024))
            return pdf_bytes
        except ImportError:
            logger.warning("WeasyPrint not installed — PDF export unavailable")
            return None
        except Exception as exc:
            logger.error("PDF rendering failed", error=str(exc))
            return None