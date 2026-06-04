"""
Dynamic LangGraph Orchestrator
───────────────────────────────
Replaces the deterministic pipeline with a dynamic LangGraph agent.
The LLM decides which agents to call based on the question.
"""

from __future__ import annotations
import time
import json
import pandas as pd
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
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


# ── State ─────────────────────────────────────────────────────────────────────

class BIState(TypedDict):
    question: str
    intent: dict
    sql: str
    data: list
    data_columns: list
    analysis: dict
    charts: list
    insights: list
    report: dict
    steps: list
    error: str | None


# ── Node Functions ────────────────────────────────────────────────────────────

def parse_intent_node(state: BIState, llm_config: dict) -> BIState:
    logger.info("▶ query_understanding")
    agent = QueryUnderstandingAgent(llm_config)
    intent = agent.parse(state["question"])
    return {**state, "intent": intent, "steps": state["steps"] + ["query_understanding"]}


def generate_sql_node(state: BIState, llm_config: dict) -> BIState:
    logger.info("▶ sql_generation")
    agent = SQLGeneratorAgent(llm_config)
    sql = agent.generate(state["intent"])
    return {**state, "sql": sql, "steps": state["steps"] + ["sql_generation"]}


def execute_sql_node(state: BIState) -> BIState:
    logger.info("▶ query_execution")
    try:
        sql = state["sql"]
        with get_session() as session:
            rows = session.execute(text(sql))
            df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
        logger.info("Query executed", rows=len(df))
        return {
            **state,
            "data": df.to_dict("records"),
            "data_columns": list(df.columns),
            "steps": state["steps"] + ["query_execution"]
        }
    except Exception as exc:
        logger.warning("SQL failed, asking LLM to fix it", error=str(exc))
        try:
            from langchain_ollama import OllamaLLM
            from langchain.schema import HumanMessage
            llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434", temperature=0)
            fix_prompt = f"""The following PostgreSQL query failed:

{state['sql']}

Error: {str(exc)}

Fix the SQL. Common fixes:
- ROUND(double precision, int) → ROUND(CAST(... AS NUMERIC), 2)
- Window functions need subquery if using GROUP BY

Return ONLY the fixed SQL query, no explanation."""
            fixed_sql = str(llm.invoke([HumanMessage(content=fix_prompt)]))
            import re
            fixed_sql = re.sub(r"```(?:sql)?|```", "", fixed_sql).strip()
            if not fixed_sql.endswith(";"):
                fixed_sql += ";"
            with get_session() as session:
                rows = session.execute(text(fixed_sql))
                df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
            logger.info("Fixed SQL executed", rows=len(df))
            return {
                **state,
                "sql": fixed_sql,
                "data": df.to_dict("records"),
                "data_columns": list(df.columns),
                "steps": state["steps"] + ["query_execution"]
            }
        except Exception as exc2:
            logger.error("SQL execution failed", error=str(exc2))
            return {**state, "error": str(exc2), "steps": state["steps"] + ["query_execution"]}


def analyze_data_node(state: BIState) -> BIState:
    logger.info("▶ data_analysis")
    if not state.get("data"):
        return state
    df = pd.DataFrame(state["data"])
    agent = DataAnalystAgent()
    analysis = agent.analyze(df, state["intent"])
    return {**state, "analysis": analysis, "steps": state["steps"] + ["data_analysis"]}


def visualize_node(state: BIState) -> BIState:
    logger.info("▶ visualization")
    if not state.get("data"):
        return state
    df = pd.DataFrame(state["data"])
    agent = VisualizationAgent()
    charts = agent.generate(df, state.get("analysis", {}), state["intent"])
    charts_serializable = [
        {"title": c["title"], "figure_json": c["figure_json"]}
        for c in charts
    ]
    return {**state, "charts": charts_serializable, "steps": state["steps"] + ["visualization"]}


def generate_insights_node(state: BIState, llm_config: dict) -> BIState:
    logger.info("▶ insight_generation")
    agent = InsightGeneratorAgent(llm_config)
    insights = agent.generate(
        state.get("analysis", {}),
        state.get("intent", {}),
        state["question"]
    )
    return {**state, "insights": insights, "steps": state["steps"] + ["insight_generation"]}


def write_report_node(state: BIState) -> BIState:
    logger.info("▶ report_writing")
    agent = ReportWriterAgent()
    
    # Rebuild full charts with PNG for PDF
    charts_for_report = []
    if state.get("data"):
        df = pd.DataFrame(state["data"])
        viz_agent = VisualizationAgent()
        full_charts = viz_agent.generate(df, state.get("analysis", {}), state.get("intent", {}))
        charts_for_report = full_charts

    report = agent.compile(
        question=state["question"],
        intent=state.get("intent", {}),
        analysis=state.get("analysis", {}),
        insights=state.get("insights", []),
        charts=charts_for_report,
        sql_query=state.get("sql", ""),
    )
    return {**state, "report": report, "steps": state["steps"] + ["report_writing"]}


# ── Router — decides what to do next ─────────────────────────────────────────

def should_analyze(state: BIState) -> str:
    """After SQL execution, decide next step."""
    if state.get("error"):
        return END
    if not state.get("data"):
        return END
    return "analyze"


def should_visualize(state: BIState) -> str:
    """After analysis, always visualize."""
    return "visualize"


def should_generate_insights(state: BIState) -> str:
    """After visualization, always generate insights."""
    return "insights"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph(llm_config: dict):
    """Build the dynamic LangGraph pipeline."""

    workflow = StateGraph(BIState)

    # Add nodes with llm_config injected
    workflow.add_node("parse_intent", lambda s: parse_intent_node(s, llm_config))
    workflow.add_node("generate_sql", lambda s: generate_sql_node(s, llm_config))
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("analyze", analyze_data_node)
    workflow.add_node("visualize", visualize_node)
    workflow.add_node("insights", lambda s: generate_insights_node(s, llm_config))
    workflow.add_node("report", write_report_node)

    # Set entry point
    workflow.set_entry_point("parse_intent")

    # Define edges
    workflow.add_edge("parse_intent", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_conditional_edges("execute_sql", should_analyze, {
        "analyze": "analyze",
        END: END
    })
    workflow.add_conditional_edges("analyze", should_visualize, {
        "visualize": "visualize"
    })
    workflow.add_conditional_edges("visualize", should_generate_insights, {
        "insights": "insights"
    })
    workflow.add_edge("insights", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ── Orchestrator ──────────────────────────────────────────────────────────────

class BIOrchestrator:
    """Dynamic LangGraph-based BI pipeline."""

    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        self.graph = build_graph(llm_config)
        logger.info("BIOrchestrator initialized — dynamic LangGraph mode")

    def run(self, question: str) -> dict:
        pipeline_start = time.time()

        initial_state: BIState = {
            "question": question,
            "intent": {},
            "sql": "",
            "data": [],
            "data_columns": [],
            "analysis": {},
            "charts": [],
            "insights": [],
            "report": {},
            "steps": [],
            "error": None,
        }

        try:
            final_state = self.graph.invoke(initial_state)
            elapsed = round(time.time() - pipeline_start, 2)

            report = final_state.get("report", {})

            logger.info(
                "Pipeline complete",
                elapsed=elapsed,
                rows=len(final_state.get("data", [])),
                charts=len(final_state.get("charts", [])),
                insights=len(final_state.get("insights", [])),
            )

            return {
                "status": "success" if not final_state.get("error") else "error",
                "question": question,
                "intent": final_state.get("intent", {}),
                "sql": final_state.get("sql", ""),
                "data_columns": final_state.get("data_columns", []),
                "analysis": {
                    k: v for k, v in final_state.get("analysis", {}).items()
                    if k not in ("trend_data", "comparison_table")
                },
                "charts": final_state.get("charts", []),
                "insights": final_state.get("insights", []),
                "report_markdown": report.get("markdown", ""),
                "kpis": report.get("kpis", []),
                "pdf_available": report.get("pdf_bytes") is not None,
                "_pdf_bytes": report.get("pdf_bytes"),
                "elapsed_seconds": elapsed,
                "steps": [{"name": s, "status": "ok", "elapsed_s": 0} for s in final_state.get("steps", [])],
                "error": final_state.get("error"),
            }

        except Exception as exc:
            logger.error("Pipeline failed", error=str(exc), exc_info=True)
            return {
                "status": "error",
                "question": question,
                "error": str(exc),
                "steps": [],
            }