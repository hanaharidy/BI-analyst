"""
Orchestrator
─────────────
The master coordinator that runs the full multi-agent pipeline.
Each step is logged with timing so you can see exactly where time is spent.
"""

from __future__ import annotations
import time
import pandas as pd
from sqlalchemy import text
from backend.db.connection import get_session
from backend.agents.query_agent import QueryUnderstandingAgent
from backend.agents.sql_agent import SQLGeneratorAgent
from backend.agents.analyst_agent import DataAnalystAgent
from backend.agents.viz_agent import VisualizationAgent
from backend.agents.insight_agent import InsightGeneratorAgent
from backend.agents.report_agent import ReportWriterAgent
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BIOrchestrator:
    """
    Runs the full pipeline:
    Question → Intent → SQL → Data → Analysis → Charts → Insights → Report
    """

    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        self.query_agent = QueryUnderstandingAgent(llm_config)
        self.sql_agent = SQLGeneratorAgent(llm_config)
        self.analyst = DataAnalystAgent()
        self.viz_agent = VisualizationAgent()
        self.insight_agent = InsightGeneratorAgent(llm_config)
        self.report_agent = ReportWriterAgent()
        logger.info("BIOrchestrator initialized")

    def run(self, question: str) -> dict:
        """
        Full pipeline execution.
        Returns a rich result dict consumed by the Streamlit frontend.
        """
        pipeline_start = time.time()
        result = {
            "question": question,
            "status": "running",
            "steps": [],
            "error": None,
        }

        try:
            # ── Step 1: Parse Intent ─────────────────────────────────────────
            intent = self._timed_step(result, "query_understanding", lambda: (
                self.query_agent.parse(question)
            ))

            # ── Step 2: Generate SQL ─────────────────────────────────────────
            sql = self._timed_step(result, "sql_generation", lambda: (
                self.sql_agent.generate(intent)
            ))

            # ── Step 3: Execute Query ────────────────────────────────────────
            df = self._timed_step(result, "query_execution", lambda: (
                self._execute_sql(sql)
            ))

            if df.empty:
                result["status"] = "no_data"
                result["error"] = "The query returned no results. Try a broader question."
                return result

            # ── Step 4: Statistical Analysis ────────────────────────────────
            analysis = self._timed_step(result, "data_analysis", lambda: (
                self.analyst.analyze(df, intent)
            ))

            # ── Step 5: Visualizations ───────────────────────────────────────
            charts = self._timed_step(result, "visualization", lambda: (
                self.viz_agent.generate(df, analysis, intent)
            ))

            # ── Step 6: Insights ─────────────────────────────────────────────
            insights = self._timed_step(result, "insight_generation", lambda: (
                self.insight_agent.generate(analysis, intent, question)
            ))

            # ── Step 7: Report ───────────────────────────────────────────────
            report = self._timed_step(result, "report_writing", lambda: (
                self.report_agent.compile(question, intent, analysis, insights, charts, sql)
            ))

            # ── Assemble final result ────────────────────────────────────────
            result.update({
                "status": "success",
                "intent": intent,
                "sql": sql,
                "data": df.to_dict("records"),
                "data_columns": list(df.columns),
                "analysis": {
                    k: v for k, v in analysis.items()
                    if k not in ("trend_data", "comparison_table")  # large, in charts
                },
                "charts": [
                    {"title": c["title"], "figure_json": c["figure_json"]}
                    for c in charts
                ],
                "insights": insights,
                "report_markdown": report["markdown"],
                "kpis": report["kpis"],
                "pdf_available": report["pdf_bytes"] is not None,
                "_pdf_bytes": report["pdf_bytes"],  # internal, not serialized to JSON
                "elapsed_seconds": round(time.time() - pipeline_start, 2),
            })

            logger.info(
                "Pipeline complete",
                elapsed=result["elapsed_seconds"],
                rows=len(df),
                charts=len(charts),
                insights=len(insights),
            )

        except Exception as exc:
            logger.error("Pipeline failed", error=str(exc), exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _timed_step(self, result: dict, step_name: str, fn) -> any:
        """Execute a pipeline step with timing and error capture."""
        start = time.time()
        logger.info(f"▶ {step_name}")
        try:
            output = fn()
            elapsed = round(time.time() - start, 3)
            result["steps"].append({"name": step_name, "status": "ok", "elapsed_s": elapsed})
            logger.info(f"✓ {step_name}", elapsed_s=elapsed)
            return output
        except Exception as exc:
            elapsed = round(time.time() - start, 3)
            result["steps"].append({"name": step_name, "status": "error", "elapsed_s": elapsed, "error": str(exc)})
            logger.error(f"✗ {step_name} failed", error=str(exc))
            raise

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """Run SQL with auto-retry on failure — LLM fixes its own mistakes."""
        sql = self._fix_sql(sql)
        try:
            with get_session() as session:
                rows = session.execute(text(sql))
                df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
            logger.info("Query executed", rows=len(df))
            return df
        except Exception as exc:
            logger.warning("SQL failed, asking LLM to fix it", error=str(exc))
            fixed_sql = self._ask_llm_to_fix_sql(sql, str(exc))
            with get_session() as session:
                rows = session.execute(text(fixed_sql))
                df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
            logger.info("Fixed SQL executed", rows=len(df))
            return df

    def _ask_llm_to_fix_sql(self, broken_sql: str, error: str) -> str:
        """Ask the LLM to fix a broken SQL query given the error message."""
        from langchain.schema import HumanMessage, SystemMessage
        from backend.db.connection import get_schema_description

        prompt = f"""You are a PostgreSQL expert. Fix this SQL query that failed.

ERROR:
{error}

BROKEN SQL:
{broken_sql}

SCHEMA:
{get_schema_description()}

Rules:
- Return ONLY the fixed SQL, no explanation, no markdown
- CTE columns cannot be referenced in outer SELECT unless included in the CTE's SELECT list
- Never use DATE_TRUNC on an already-aliased column

Fixed SQL:"""

        messages = [
            SystemMessage(content="You fix broken PostgreSQL queries. Return only valid SQL."),
            HumanMessage(content=prompt),
        ]
        response = self.sql_agent.llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        fixed = self.sql_agent._clean_sql(raw)
        logger.info("LLM fixed SQL", length=len(fixed))
        return fixed

    def _fix_sql(self, sql: str) -> str:
        """Fix common LLM CTE mistakes before execution."""
        import re

        # The core bug: LLM puts 'quarter' in order_data CTE but not in region_data,
        # then tries to SELECT it from region_data. Fix: remove it from final SELECT.
        # Replace "quarter AS quarter," or "quarter AS quarter" in final SELECT
        sql = re.sub(r"\bquarter\s+AS\s+quarter\s*,?\s*\n", "\n", sql)

        # Also fix DATE_TRUNC on already-aliased CTE columns
        sql = re.sub(
            r"DATE_TRUNC\('[^']+',\s*(quarter|period|month|week)\)",
            r"\1",
            sql
        )

        # Fix: if region_data doesn't include quarter, add it to region_data CTE
        if "region_data" in sql and "quarter" in sql:
            sql = re.sub(
                r"(SELECT\s+region,\s+SUM\(revenue\)\s+AS\s+total_revenue\s+FROM\s+order_data\s+GROUP BY\s+region)",
                "SELECT region, quarter, SUM(revenue) AS total_revenue FROM order_data GROUP BY region, quarter",
                sql,
                flags=re.IGNORECASE | re.DOTALL
            )

        return sql