"""
Visualization Agent
────────────────────
Generates Plotly figures from DataFrames and analysis results.
Returns figures as JSON (for Streamlit) and PNG bytes (for PDF reports).
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Any
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent color palette across all charts
COLORS = {
    "primary": "#2563EB",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#6B7280",
    "palette": px.colors.qualitative.Set2,
}

LAYOUT_DEFAULTS = dict(
    font_family="'Inter', sans-serif",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#F3F4F6", gridwidth=1),
)


class VisualizationAgent:
    """Creates Plotly charts based on data and analysis context."""

    def generate(self, df: pd.DataFrame, analysis: dict, intent: dict) -> list[dict]:
        """
        Returns a list of chart dicts, each with:
          - 'title': str
          - 'figure_json': str  (Plotly JSON for Streamlit)
          - 'figure_png': bytes (for PDF embedding)
        """
        if df.empty:
            logger.warning("No data to visualize")
            return []

        charts = []
        analysis_type = intent.get("analysis_type", "summary")

        logger.info("Generating visualizations", analysis_type=analysis_type)

        if analysis_type == "trend":
            charts.extend(self._trend_charts(df, analysis, intent))
        elif analysis_type == "ranking":
            charts.extend(self._ranking_charts(df, analysis, intent))
        elif analysis_type == "comparison":
            charts.extend(self._comparison_charts(df, analysis, intent))
        elif analysis_type == "forecast":
            charts.extend(self._forecast_charts(df, analysis, intent))
        else:
            # Summary: always show a bar + optional trend
            charts.extend(self._ranking_charts(df, analysis, intent))
            if "trend_data" in analysis:
                charts.extend(self._trend_charts(df, analysis, intent))

        logger.info("Charts generated", count=len(charts))
        return charts

    # ── Chart Builders ────────────────────────────────────────────────────────

    def _trend_charts(self, df: pd.DataFrame, analysis: dict, intent: dict) -> list[dict]:
        charts = []
        trend_data = analysis.get("trend_data")
        if not trend_data:
            return charts

        tdf = pd.DataFrame(trend_data)
        metric = analysis.get("primary_metric", "revenue")
        period_col = self._find_period_col(tdf)
        if not period_col:
            return charts

        # Line chart with moving average
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tdf[period_col], y=tdf[metric],
            mode="lines+markers",
            name=metric.replace("_", " ").title(),
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=7),
        ))
        ma_col = f"ma_3"
        if ma_col in tdf.columns:
            fig.add_trace(go.Scatter(
                x=tdf[period_col], y=tdf[ma_col],
                mode="lines", name="3-Period MA",
                line=dict(color=COLORS["warning"], width=2, dash="dash"),
            ))

        # Overlay forecast
        forecast = analysis.get("forecast_next_3", [])
        if forecast:
            last_period = tdf[period_col].iloc[-1]
            forecast_x = [f"Forecast +{i+1}" for i in range(len(forecast))]
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast,
                mode="lines+markers", name="Forecast",
                line=dict(color=COLORS["success"], width=2, dash="dot"),
                marker=dict(symbol="diamond", size=8),
            ))

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Trend Over Time",
            **LAYOUT_DEFAULTS,
        )
        charts.append(self._package(f"{metric}_trend", fig))

        # Growth rate bar chart
        if "growth_rate_pct" in tdf.columns:
            gr = tdf.dropna(subset=["growth_rate_pct"])
            colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in gr["growth_rate_pct"]]
            fig2 = go.Figure(go.Bar(
                x=gr[period_col], y=gr["growth_rate_pct"],
                marker_color=colors,
                name="Growth %",
            ))
            fig2.add_hline(y=0, line_dash="solid", line_color=COLORS["muted"])
            fig2.update_layout(title="Period-over-Period Growth Rate (%)", **LAYOUT_DEFAULTS)
            charts.append(self._package("growth_rate", fig2))

        return charts

    def _ranking_charts(self, df: pd.DataFrame, analysis: dict, intent: dict) -> list[dict]:
        charts = []
        rankings = analysis.get("rankings")
        dim_col = analysis.get("dimension_used")
        metric = analysis.get("primary_metric", "revenue")

        if not dim_col or not metric:
            # Fallback: just aggregate the first string and numeric col
            str_cols = df.select_dtypes(include="object").columns
            num_cols = df.select_dtypes(include="number").columns
            if str_cols.empty or num_cols.empty:
                return charts
            dim_col, metric = str_cols[0], num_cols[0]

        # Horizontal bar — sorted
        grouped = df.groupby(dim_col)[metric].sum().sort_values(ascending=True).reset_index()
        avg = grouped[metric].mean()

        colors = [
            COLORS["danger"] if v < avg * 0.8
            else COLORS["warning"] if v < avg
            else COLORS["success"]
            for v in grouped[metric]
        ]

        fig = go.Figure(go.Bar(
            x=grouped[metric], y=grouped[dim_col],
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}" if v > 1000 else f"{v:,.0f}" for v in grouped[metric]],
            textposition="outside",
        ))
        fig.add_vline(x=avg, line_dash="dash", line_color=COLORS["muted"],
                      annotation_text=f"Avg: ${avg:,.0f}")
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} by {dim_col.replace('_', ' ').title()}",
            **LAYOUT_DEFAULTS,
        )
        charts.append(self._package(f"ranking_{dim_col}", fig))

        # Pie chart for share
        if len(grouped) <= 10:
            fig2 = go.Figure(go.Pie(
                labels=grouped[dim_col], values=grouped[metric],
                hole=0.4,
                marker_colors=COLORS["palette"],
            ))
            fig2.update_layout(title=f"Share of {metric.replace('_', ' ').title()}", **LAYOUT_DEFAULTS)
            charts.append(self._package(f"share_{dim_col}", fig2))

        return charts

    def _comparison_charts(self, df: pd.DataFrame, analysis: dict, intent: dict) -> list[dict]:
        comparison = analysis.get("comparison_table")
        if not comparison:
            return self._ranking_charts(df, analysis, intent)

        cdf = pd.DataFrame(comparison)
        dim_col = cdf.columns[0]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cdf[dim_col], y=cdf["total"],
            name="Total", marker_color=COLORS["primary"],
        ))
        overall_mean = analysis.get("overall_mean", 0)
        fig.add_hline(y=overall_mean, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"Mean: ${overall_mean:,.0f}")
        fig.update_layout(title="Comparative Performance", **LAYOUT_DEFAULTS)
        return [self._package("comparison", fig)]

    def _forecast_charts(self, df: pd.DataFrame, analysis: dict, intent: dict) -> list[dict]:
        hist = analysis.get("historical_data", [])
        forecast = analysis.get("forecast_6_periods", [])
        if not hist:
            return []

        hdf = pd.DataFrame(hist)
        period_col = self._find_period_col(hdf)
        metric = hdf.columns[1] if len(hdf.columns) > 1 else None
        if not period_col or not metric:
            return []

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hdf[period_col].astype(str), y=hdf[metric],
            mode="lines+markers", name="Historical",
            line=dict(color=COLORS["primary"], width=3),
        ))
        if forecast:
            forecast_x = [f"+{i+1}" for i in range(len(forecast))]
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast,
                mode="lines+markers", name="Forecast",
                line=dict(color=COLORS["success"], width=2, dash="dot"),
                marker=dict(symbol="diamond"),
            ))
        fig.update_layout(title="Forecast (Holt-Winters)", **LAYOUT_DEFAULTS)
        return [self._package("forecast", fig)]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _find_period_col(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if any(k in col.lower() for k in ["period", "date", "month", "quarter"]):
                return col
        return None

    def _package(self, name: str, fig: go.Figure) -> dict:
        """Convert a figure to JSON (Streamlit) + PNG bytes (PDF)."""
        try:
            png_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
        except Exception:
            png_bytes = None  # kaleido may not be available in all envs

        return {
            "title": name.replace("_", " ").title(),
            "figure_json": fig.to_json(),
            "figure_png": png_bytes,
        }