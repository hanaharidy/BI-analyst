"""
Streamlit Frontend — AI Business Intelligence Analyst
──────────────────────────────────────────────────────
Clean, executive-grade dashboard for querying data in plain English.
Results are stored in session_state so PDF download never wipes the page.
"""

import json
import time
import streamlit as st
import plotly.graph_objects as go
import httpx
from datetime import datetime

API_BASE = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

.kpi-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #1E40AF 100%);
    color: white; border-radius: 12px; padding: 20px 24px;
    text-align: center; margin-bottom: 8px;
}
.kpi-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.75; }
.kpi-value { font-size: 28px; font-weight: 700; margin-top: 4px; }

.insight-card {
    padding: 14px 18px; border-radius: 10px; margin-bottom: 10px;
    border-left: 4px solid #ccc;
}
.insight-positive    { background: #F0FDF4; border-color: #10B981; }
.insight-warning     { background: #FFFBEB; border-color: #F59E0B; }
.insight-risk        { background: #FEF2F2; border-color: #EF4444; }
.insight-opportunity { background: #F5F3FF; border-color: #6366F1; }
.insight-title  { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
.insight-body   { font-size: 13px; color: #374151; line-height: 1.6; }
.insight-action { font-size: 12px; color: #6B7280; font-style: italic; margin-top: 6px; }

.sql-block {
    background: #1E293B; color: #94A3B8; padding: 16px; border-radius: 8px;
    font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.6;
    overflow-x: auto; white-space: pre;
}
</style>
""", unsafe_allow_html=True)

# ── Example Questions ─────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Show me revenue trends by region for Q4 2025",
    "Which products are underperforming this quarter?",
    "Compare sales across customer segments year over year",
    "Top 5 customers by total revenue in 2025",
    "What is the revenue forecast for next quarter?",
    "Show monthly profit margin trends for Electronics category",
    "Which regions have the highest growth rate?",
    "Detect any revenue anomalies in 2025",
]

# ── API Helpers ───────────────────────────────────────────────────────────────

def call_api(question: str) -> dict:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_BASE}/analyze", json={"question": question})
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"status": "error", "error": "Cannot connect to backend. Is uvicorn running on port 8000?"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def fetch_pdf(question: str) -> bytes | None:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_BASE}/report/pdf", json={"question": question})
            if resp.status_code == 200:
                return resp.content
    except Exception:
        pass
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 BI Analyst")
    st.markdown("*Ask business questions in plain English*")
    st.divider()
    st.markdown("#### 💡 Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"eg_{q[:20]}", use_container_width=True):
            st.session_state["current_question"] = q
            # Clear previous results when a new question is chosen
            st.session_state.pop("result", None)
            st.session_state.pop("cached_pdf", None)
    st.divider()
    st.markdown("#### ℹ️ About")
    st.caption("Multi-agent pipeline: Query Understanding → SQL → Analysis → Visualization → Insights → Report")
    st.caption("Powered by LangChain + Llama 3.1 + PostgreSQL")

# ── Main Layout ───────────────────────────────────────────────────────────────

st.title("🧠 AI Business Intelligence Analyst")
st.caption("Ask any business question — the AI pipeline handles the rest.")

default_q = st.session_state.get("current_question", "")
question = st.text_input(
    "Your question",
    value=default_q,
    placeholder="e.g. Show me revenue trends by region for Q4 2025",
    label_visibility="collapsed",
)

run_btn = st.button("Analyze →", type="primary")

# ── Run Pipeline ──────────────────────────────────────────────────────────────

if run_btn and question.strip():
    # Clear stale results from previous question
    st.session_state.pop("result", None)
    st.session_state.pop("cached_pdf", None)

    with st.status("Running analysis pipeline...", expanded=True) as status:
        st.write("🔍 Parsing intent...")
        start = time.time()
        result = call_api(question)
        elapsed = round(time.time() - start, 1)

        if result.get("status") == "success":
            status.update(label=f"✅ Analysis complete ({elapsed}s)", state="complete")
            # Fetch PDF immediately — inside the same run, before any rerun
            if result.get("pdf_available"):
                st.session_state["cached_pdf"] = fetch_pdf(question)
        else:
            status.update(label="❌ Pipeline failed", state="error")

    # Persist result so download_button rerun doesn't erase it
    st.session_state["result"] = result
    st.session_state["current_question"] = question

elif run_btn:
    st.warning("Please enter a question.")

# ── Render Results (always from session_state) ────────────────────────────────

result = st.session_state.get("result")

if not result:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #9CA3AF;">
        <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
        <div style="font-size: 18px; font-weight: 600; color: #374151; margin-bottom: 8px;">
            Ask a business question to get started
        </div>
        <div style="font-size: 14px;">
            Try: "Show me revenue trends by region for Q4 2025"
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if result.get("status") == "error":
    st.error(f"**Error:** {result.get('error')}")
    st.stop()

if result.get("status") == "no_data":
    st.warning(result.get("error", "No data returned."))
    st.stop()

# ── Pipeline Steps ────────────────────────────────────────────────────────────

with st.expander("Pipeline trace", expanded=False):
    for step in result.get("steps", []):
        icon = "✅" if step["status"] == "ok" else "❌"
        st.markdown(
            f"{icon} **{step['name'].replace('_', ' ').title()}** — {step['elapsed_s']}s"
            + (f" — `{step.get('error')}`" if step.get("error") else "")
        )

# ── KPI Cards ─────────────────────────────────────────────────────────────────

kpis = result.get("kpis", [])
if kpis:
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{kpi["label"]}</div>'
                f'<div class="kpi-value">{kpi["value"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ── Charts + Insights ─────────────────────────────────────────────────────────

chart_col, insight_col = st.columns([3, 2])

with chart_col:
    st.subheader("📈 Visualizations")
    charts = result.get("charts", [])
    if charts:
        tabs = st.tabs([c["title"] for c in charts])
        for tab, chart in zip(tabs, charts):
            with tab:
                fig = go.Figure(json.loads(chart["figure_json"]))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No visualizations generated.")

with insight_col:
    st.subheader("💡 Insights")
    type_icon = {"positive": "✅", "warning": "⚠️", "risk": "🔴", "opportunity": "🟣"}
    for ins in result.get("insights", []):
        icon = type_icon.get(ins.get("type", ""), "💡")
        css_class = f"insight-{ins.get('type', 'positive')}"
        st.markdown(
            f'<div class="insight-card {css_class}">'
            f'<div class="insight-title">{icon} {ins["title"]}</div>'
            f'<div class="insight-body">{ins["body"]}</div>'
            f'<div class="insight-action">→ {ins["action"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Report + SQL ──────────────────────────────────────────────────────────────

report_col, sql_col = st.columns([1, 1])

with report_col:
    st.subheader("📄 Executive Summary")
    st.markdown(result.get("report_markdown", ""))

    # PDF is pre-fetched in session_state — download_button won't trigger a rerun
    cached_pdf = st.session_state.get("cached_pdf")
    if cached_pdf:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=cached_pdf,
            file_name=f"bi_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

with sql_col:
    st.subheader("🔍 Generated SQL")
    sql = result.get("sql", "")
    if sql:
        st.markdown(f'<div class="sql-block">{sql}</div>', unsafe_allow_html=True)

    intent = result.get("intent", {})
    if intent:
        with st.expander("Parsed Intent"):
            st.json(intent)