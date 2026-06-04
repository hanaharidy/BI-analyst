"""
Data Analyst Agent (LLM‑Driven)
────────────────────────────────
Uses LLM to decide which statistical analyses to run based on data shape,
user intent, and the original question — no hardcoded if/else on analysis_type.
"""

import pandas as pd
import numpy as np
import json
from typing import Any
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage
from backend.tools.stats_tools import (
    compute_summary_stats,
    detect_anomalies,
    compute_growth_rates,
    compute_moving_average,
    forecast_next_periods,
    rank_performance,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior data analyst. Given a DataFrame description and a business question, decide which analyses to perform.

Available analysis functions (you can select one or more):
- trend: time‑series growth rates, moving average, forecast
- ranking: sort groups by a metric, identify top/bottom performers
- comparison: compare groups against overall average
- anomaly: statistical outlier detection (Z‑score)
- forecast: forward prediction using Holt‑Winters
- summary: basic stats (mean, total, min, max) – always include

Return a JSON object with:
{
  "analysis_plan": ["trend", "ranking"],   // list of analyses to run
  "parameters": {
    "metric_priority": ["revenue", "profit"],  // optional ordering
    "dimension_hint": "region",                // optional
    "period_hint": "order_date"                // optional
  }
}

Only request analyses that make sense for the data shape and the question.
If the data has no time column → do NOT request trend or forecast.
If data has only one row → do NOT request ranking or comparison.
"""

class DataAnalystAgent:
    """Performs statistical analysis with LLM‑decided strategy."""

    def __init__(self, llm_config: dict):
        self.llm = self._build_llm(llm_config)
        logger.info("LLM‑driven DataAnalystAgent ready")

    def _build_llm(self, cfg: dict):
        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0,
        )

    def analyze(self, df: pd.DataFrame, intent: dict, question: str = "") -> dict[str, Any]:
        """Run appropriate analysis based on LLM decision."""
        if df.empty:
            logger.warning("Empty DataFrame — skipping analysis")
            return {"error": "No data returned for this query."}

        # Basic info always included
        analysis = {
            "row_count": len(df),
            "columns": list(df.columns),
            "analysis_type": intent.get("analysis_type", "summary"),
        }

        # Detect numeric columns and potential dimensions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()
        primary_metric = self._pick_primary_metric(numeric_cols)
        if primary_metric:
            analysis["summary_stats"] = compute_summary_stats(df, primary_metric)
            analysis["primary_metric"] = primary_metric

        # ── LLM decides the analysis plan ─────────────────────────────────
        plan = self._get_analysis_plan(df, intent, question, numeric_cols, string_cols)
        logger.info("LLM analysis plan", plan=plan)

        for analysis_type in plan.get("analysis_plan", []):
            if analysis_type == "trend":
                analysis.update(self._trend_analysis(df, primary_metric))
            elif analysis_type == "ranking":
                analysis.update(self._ranking_analysis(df, intent, primary_metric))
            elif analysis_type == "comparison":
                analysis.update(self._comparison_analysis(df, intent, primary_metric))
            elif analysis_type == "anomaly":
                analysis.update(self._anomaly_analysis(df, primary_metric))
            elif analysis_type == "forecast":
                analysis.update(self._forecast_analysis(df, primary_metric))
            # summary is already added above

        return analysis

    def _get_analysis_plan(self, df: pd.DataFrame, intent: dict, question: str,
                           numeric_cols: list, string_cols: list) -> dict:
        """Call LLM to decide which analyses to run."""
        # Prepare compact description of the data
        data_desc = {
            "shape": df.shape,
            "numeric_columns": numeric_cols,
            "string_columns": string_cols,
            "has_date_column": any("date" in c.lower() or "period" in c.lower() for c in df.columns),
            "row_count": len(df),
            "unique_values_per_string_col": {c: min(5, df[c].nunique()) for c in string_cols[:3]},
        }

        user_prompt = f"""
Business question: {question or intent.get('intent_summary', 'Not provided')}

Intent hint from previous agent: {intent.get('analysis_type', 'summary')}

Data description:
{json.dumps(data_desc, indent=2)}

Decide which analyses to run. Return JSON only.
"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            plan = json.loads(raw)
            # Ensure analysis_plan is a list
            if "analysis_plan" not in plan:
                plan["analysis_plan"] = [intent.get("analysis_type", "summary")]
            return plan
        except Exception as e:
            logger.warning(f"LLM plan failed, using fallback: {e}")
            # Fallback: use intent hint
            return {"analysis_plan": [intent.get("analysis_type", "summary")], "parameters": {}}

    # ── Analysis Strategies ───────────────────────────────────────────────

    def _trend_analysis(self, df: pd.DataFrame, metric: str) -> dict:
        """Time‑series trend: growth rates, moving average, forecast."""
        period_col = self._find_period_column(df)
        if not period_col or not metric:
            return {}
        df_sorted = df.copy().sort_values(period_col)
        df_sorted[period_col] = df_sorted[period_col].astype(str)
        period_totals = df_sorted.groupby(period_col)[metric].sum().reset_index()
        period_totals = compute_growth_rates(period_totals, period_col, metric)
        period_totals = compute_moving_average(period_totals, metric, window=3)
        forecast = forecast_next_periods(period_totals[metric], periods=3)
        return {
            "trend_data": period_totals.to_dict("records"),
            "forecast_next_3": forecast,
            "avg_growth_rate": round(period_totals["growth_rate_pct"].dropna().mean(), 2),
            "total_growth": self._total_growth(period_totals[metric]),
        }

    def _ranking_analysis(self, df: pd.DataFrame, intent: dict, metric: str) -> dict:
        """Rank groups by metric — identifies stars and laggards."""
        dim_col = self._find_dimension_column(df, intent)
        if not dim_col or not metric:
            return {}
        rankings = rank_performance(df, dim_col, metric, top_n=5)
        return {"rankings": rankings, "dimension_used": dim_col}

    def _comparison_analysis(self, df: pd.DataFrame, intent: dict, metric: str) -> dict:
        """Compare groups against each other and the mean."""
        dim_col = self._find_dimension_column(df, intent)
        if not dim_col or not metric:
            return {}
        grouped = df.groupby(dim_col)[metric].agg(["sum", "mean", "count"]).round(2)
        grouped.columns = ["total", "avg", "count"]
        overall_mean = grouped["total"].mean()
        grouped["vs_avg_pct"] = ((grouped["total"] - overall_mean) / overall_mean * 100).round(1)
        return {
            "comparison_table": grouped.reset_index().to_dict("records"),
            "overall_mean": round(overall_mean, 2),
        }

    def _anomaly_analysis(self, df: pd.DataFrame, metric: str) -> dict:
        """Detect statistical anomalies in the metric."""
        if not metric:
            return {}
        df_flagged = detect_anomalies(df, metric)
        anomalies = df_flagged[df_flagged["is_anomaly"]].to_dict("records")
        return {"anomalies": anomalies, "anomaly_count": len(anomalies)}

    def _forecast_analysis(self, df: pd.DataFrame, metric: str) -> dict:
        """Generate forward‑looking forecasts and return historical data for visualisation."""
        period_col = self._find_period_column(df)
        if not period_col or not metric:
            return {}

        # Aggregate by period (e.g., month, quarter)
        period_totals = df.groupby(period_col)[metric].sum().sort_index()

        # Store historical data as list of dicts
        historical = period_totals.reset_index().rename(
            columns={period_col: "period", metric: "value"}
        ).to_dict("records")

        # Generate forecast for next 6 periods
        forecast = forecast_next_periods(period_totals, periods=6)

        return {
            "historical_data": historical,
            "forecast_6_periods": forecast,
            "primary_metric": metric,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_primary_metric(self, numeric_cols: list[str]) -> str | None:
        """Select the most likely metric column (revenue > profit > total_sales > amount > quantity)."""
        priority = ["revenue", "profit", "total_revenue", "total_sales", "amount", "quantity"]
        for name in priority:
            for col in numeric_cols:
                if name in col.lower():
                    return col
        return numeric_cols[0] if numeric_cols else None

    def _find_period_column(self, df: pd.DataFrame) -> str | None:
        """Find a column that likely represents a time period."""
        for col in df.columns:
            if any(k in col.lower() for k in ["period", "date", "month", "quarter", "week"]):
                return col
        return None

    def _find_dimension_column(self, df: pd.DataFrame, intent: dict) -> str | None:
        """Find a column that represents a grouping dimension (region, product, etc.)."""
        for dim in intent.get("dimensions", []):
            for col in df.columns:
                if dim.lower() in col.lower():
                    return col
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()
        return string_cols[0] if string_cols else None

    def _total_growth(self, series: pd.Series) -> float:
        """Calculate total growth from first to last value."""
        s = series.dropna()
        if len(s) < 2 or s.iloc[0] == 0:
            return 0.0
        return round((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100, 2)