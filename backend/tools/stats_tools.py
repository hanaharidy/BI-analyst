"""
Statistical analysis tools for the Data Analyst agent.
Wraps pandas/numpy/scipy/statsmodels into clean, agent-callable functions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Any
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def compute_summary_stats(df: pd.DataFrame, numeric_col: str) -> dict[str, Any]:
    """Basic descriptive statistics for a numeric column."""
    series = df[numeric_col].dropna()
    return {
        "mean": round(series.mean(), 2),
        "median": round(series.median(), 2),
        "std": round(series.std(), 2),
        "min": round(series.min(), 2),
        "max": round(series.max(), 2),
        "total": round(series.sum(), 2),
        "count": int(len(series)),
        "q25": round(series.quantile(0.25), 2),
        "q75": round(series.quantile(0.75), 2),
    }


def detect_anomalies(df: pd.DataFrame, value_col: str, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Z-score based anomaly detection.
    Returns a copy of df with an 'is_anomaly' boolean column.
    """
    df = df.copy()
    series = df[value_col].dropna()
    z_scores = np.abs(stats.zscore(series))
    df["z_score"] = z_scores
    df["is_anomaly"] = z_scores > z_threshold
    anomaly_count = df["is_anomaly"].sum()
    logger.info("Anomaly detection complete", anomalies_found=int(anomaly_count), column=value_col)
    return df


def compute_growth_rates(df: pd.DataFrame, period_col: str, value_col: str) -> pd.DataFrame:
    """
    Compute period-over-period growth rate (%).
    Expects df sorted by period_col ascending.
    """
    df = df.copy().sort_values(period_col)
    df["prev_value"] = df[value_col].shift(1)
    df["growth_rate_pct"] = ((df[value_col] - df["prev_value"]) / df["prev_value"] * 100).round(2)
    return df


def compute_moving_average(df: pd.DataFrame, value_col: str, window: int = 3) -> pd.DataFrame:
    """Add a rolling moving average column."""
    df = df.copy()
    df[f"ma_{window}"] = df[value_col].rolling(window=window, min_periods=1).mean().round(2)
    return df


def forecast_next_periods(series: pd.Series, periods: int = 3) -> list[float]:
    """
    Holt-Winters exponential smoothing forecast.
    Falls back to linear trend if series is too short.
    """
    series = series.dropna()

    if len(series) < 4:
        # Linear extrapolation for very short series
        x = np.arange(len(series))
        slope, intercept, *_ = stats.linregress(x, series.values)
        forecast = [round(intercept + slope * (len(series) + i), 2) for i in range(periods)]
        logger.info("Used linear forecast (series too short for Holt-Winters)")
        return forecast

    try:
        model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
        return [round(float(v), 2) for v in forecast]
    except Exception as exc:
        logger.warning("Holt-Winters failed, falling back to mean", error=str(exc))
        return [round(float(series.mean()), 2)] * periods


def rank_performance(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 5) -> dict:
    """
    Rank groups by a value and identify top/bottom performers.
    Returns dict with top_performers and bottom_performers.
    """
    ranked = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    total = ranked.sum()

    return {
        "top_performers": [
            {"name": k, "value": round(v, 2), "share_pct": round(v / total * 100, 1)}
            for k, v in ranked.head(top_n).items()
        ],
        "bottom_performers": [
            {"name": k, "value": round(v, 2), "share_pct": round(v / total * 100, 1)}
            for k, v in ranked.tail(top_n).items()
        ],
        "total": round(total, 2),
    }


def compute_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> dict:
    """Pearson correlation matrix for selected columns."""
    corr = df[columns].corr().round(3)
    return corr.to_dict()