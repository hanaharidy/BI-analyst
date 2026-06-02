"""
Data Analyst Agent
───────────────────
Runs statistical analysis on the query results DataFrame.
Computes aggregations, growth rates, anomaly detection, and forecasts.
"""

import pandas as pd
import numpy as np
from typing import Any
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


class DataAnalystAgent:
    """Performs statistical analysis on query result DataFrames."""

    def analyze(self, df: pd.DataFrame, intent: dict) -> dict[str, Any]:
        """
        Run appropriate analysis based on intent.
        Returns a rich analysis dict that downstream agents consume.
        """
        if df.empty:
            logger.warning("Empty DataFrame received — skipping analysis")
            return {"error": "No data returned for this query."}

        logger.info("Starting analysis", rows=len(df), analysis_type=intent.get("analysis_type"))

        analysis = {
            "row_count": len(df),
            "columns": list(df.columns),
            "analysis_type": intent.get("analysis_type"),
        }

        # Detect the primary numeric column (usually revenue, profit, or quantity)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        primary_metric = self._pick_primary_metric(numeric_cols)

        if primary_metric:
            analysis["summary_stats"] = compute_summary_stats(df, primary_metric)
            analysis["primary_metric"] = primary_metric

        analysis_type = intent.get("analysis_type", "summary")

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

        else:
            # summary — always run ranking and basic stats
            analysis.update(self._ranking_analysis(df, intent, primary_metric))

        logger.info("Analysis complete", keys=list(analysis.keys()))
        return analysis

    # ── Analysis Strategies ───────────────────────────────────────────────────

    def _trend_analysis(self, df: pd.DataFrame, metric: str) -> dict:
        """Time-series trend: growth rates, moving average, forecast."""
        period_col = self._find_period_column(df)
        if not period_col or not metric:
            return {}

        df_sorted = df.copy().sort_values(period_col)
        df_sorted[period_col] = df_sorted[period_col].astype(str)

        # Aggregate by period if there are dimension groups
        period_totals = df_sorted.groupby(period_col)[metric].sum().reset_index()
        period_totals = compute_growth_rates(period_totals, period_col, metric)
        period_totals = compute_moving_average(period_totals, metric, window=3)

        # Forecast next 3 periods
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
        return {
            "rankings": rankings,
            "dimension_used": dim_col,
        }

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
        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
        }

    def _forecast_analysis(self, df: pd.DataFrame, metric: str) -> dict:
        """Generate forward-looking forecasts."""
        period_col = self._find_period_column(df)
        if not period_col or not metric:
            return {}

        period_totals = df.groupby(period_col)[metric].sum().sort_index()
        forecast = forecast_next_periods(period_totals, periods=6)
        return {
            "forecast_6_periods": forecast,
            "historical_data": period_totals.reset_index().to_dict("records"),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_primary_metric(self, numeric_cols: list[str]) -> str | None:
        priority = ["revenue", "profit", "total_revenue", "total_sales", "amount", "quantity"]
        for name in priority:
            for col in numeric_cols:
                if name in col.lower():
                    return col
        return numeric_cols[0] if numeric_cols else None

    def _find_period_column(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if any(k in col.lower() for k in ["period", "date", "month", "quarter", "week"]):
                return col
        return None

    def _find_dimension_column(self, df: pd.DataFrame, intent: dict) -> str | None:
        # Try to match intent dimensions first
        for dim in intent.get("dimensions", []):
            for col in df.columns:
                if dim.lower() in col.lower():
                    return col
        # Fallback: first string column
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()
        return string_cols[0] if string_cols else None

    def _total_growth(self, series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 2 or s.iloc[0] == 0:
            return 0.0
        return round((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100, 2)