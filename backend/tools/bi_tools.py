"""
BI Tools for Supervisor Agent
──────────────────────────────
Each step of the BI pipeline is wrapped as a LangChain tool.
The Supervisor LLM decides which tool to call next based on the conversation state.
A shared dictionary holds the intermediate results.
"""

import json
import re
import pandas as pd
from sqlalchemy import text
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from backend.db.connection import get_session
from backend.agents.query_agent import QueryUnderstandingAgent
from backend.agents.sql_agent import SQLGeneratorAgent
from backend.agents.analyst_agent import DataAnalystAgent
from backend.agents.viz_agent import VisualizationAgent
from backend.agents.insight_agent import InsightGeneratorAgent
from backend.agents.report_agent import ReportWriterAgent
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# LLM configuration (shared across agents)
LLM_CONFIG = {
    "model": "llama3.1",
    "base_url": "http://localhost:11434"
}

# Global state to carry data between tool calls
shared_state = {
    "question": "",
    "intent": {},
    "sql": "",
    "data": None,
    "analysis": {},
    "charts": [],
    "insights": [],
    "report": None,
    "error": None
}


@tool
def parse_intent(question: str) -> str:
    """
    Parse the user's natural language question into a structured JSON intent.
    Call this first for any new query.
    """
    logger.info("Tool: parse_intent", question=question)
    agent = QueryUnderstandingAgent(LLM_CONFIG)
    intent = agent.parse(question)
    shared_state["question"] = question
    shared_state["intent"] = intent
    return json.dumps(intent, indent=2)


@tool
def generate_sql() -> str:
    """
    Generate a PostgreSQL query from the stored intent.
    Requires that parse_intent has been called first.
    """
    logger.info("Tool: generate_sql")
    if not shared_state["intent"]:
        return "Error: No intent found. Call parse_intent first."
    agent = SQLGeneratorAgent(LLM_CONFIG)
    sql = agent.generate(shared_state["intent"])
    shared_state["sql"] = sql
    return sql


@tool
def execute_sql() -> str:
    """
    Run the generated SQL against the PostgreSQL database.
    If it fails, ask LLM to fix it once and retry.
    Stores the result as a DataFrame in shared_state.
    """
    logger.info("Tool: execute_sql")
    if not shared_state["sql"]:
        return "Error: No SQL query found. Call generate_sql first."

    def try_execute(sql: str):
        try:
            with get_session() as session:
                rows = session.execute(text(sql))
                df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
                return True, df, None
        except Exception as e:
            return False, None, str(e)

    # First attempt
    success, df, error = try_execute(shared_state["sql"])
    if success:
        shared_state["data"] = df
        logger.info("SQL executed", rows=len(df))
        return f"Query successful. Returned {len(df)} rows. Columns: {list(df.columns)}"

    # --- Self-healing: ask LLM to fix the SQL ---
    logger.warning(f"SQL failed, asking LLM to fix. Error: {error}")
    fix_llm = ChatOllama(
        model=LLM_CONFIG["model"],
        base_url=LLM_CONFIG["base_url"],
        temperature=0
    )
    fix_prompt = f"""The following PostgreSQL query failed:

{shared_state['sql']}

Error: {error}

Fix the SQL. Return ONLY the fixed SQL query, no explanation, no markdown."""

    try:
        response = fix_llm.invoke([HumanMessage(content=fix_prompt)])
        fixed_sql = response.content if hasattr(response, "content") else str(response)
        fixed_sql = re.sub(r"```(?:sql)?|```", "", fixed_sql).strip()
        if not fixed_sql.endswith(";"):
            fixed_sql += ";"
        logger.info("LLM proposed fixed SQL", fixed_sql=fixed_sql[:200])

        success2, df2, error2 = try_execute(fixed_sql)
        if success2:
            shared_state["sql"] = fixed_sql
            shared_state["data"] = df2
            logger.info("Fixed SQL executed", rows=len(df2))
            return f"Query fixed and executed. Returned {len(df2)} rows. Columns: {list(df2.columns)}"
        else:
            shared_state["error"] = error2
            return f"SQL execution failed after LLM retry: {error2}"
    except Exception as e:
        shared_state["error"] = str(e)
        return f"Self-healing failed: {e}"


@tool
def analyze_data() -> str:
    """
    Run statistical analysis on the retrieved data.
    Requires that execute_sql has been called and data exists.
    """
    logger.info("Tool: analyze_data")
    if shared_state["data"] is None:
        return "Error: No data available. Call execute_sql first."
    agent = DataAnalystAgent(LLM_CONFIG)
    analysis = agent.analyze(shared_state["data"], shared_state["intent"], shared_state["question"])
    shared_state["analysis"] = analysis
    summary = analysis.get("summary_stats", {})
    return f"Analysis complete. Summary: total={summary.get('total')}, mean={summary.get('mean')}, rows_analyzed={analysis.get('row_count')}"


@tool
def visualize() -> str:
    """
    Generate Plotly charts based on the data and analysis.
    """
    logger.info("Tool: visualize")
    if shared_state["data"] is None:
        return "Error: No data to visualize. Call execute_sql first."
    agent = VisualizationAgent(LLM_CONFIG)
    charts = agent.generate(
        shared_state["data"],
        shared_state["analysis"],
        shared_state["intent"],
        shared_state["question"]
    )
    shared_state["charts"] = charts
    return f"Generated {len(charts)} charts: {[c['title'] for c in charts]}"


@tool
def generate_insights() -> str:
    """
    Generate executive-level insights from the statistical analysis.
    """
    logger.info("Tool: generate_insights")
    if not shared_state["analysis"]:
        return "Error: No analysis available. Call analyze_data first."
    agent = InsightGeneratorAgent(LLM_CONFIG)
    insights = agent.generate(
        shared_state["analysis"],
        shared_state["intent"],
        shared_state["question"]
    )
    shared_state["insights"] = insights
    return f"Insights generated: {len(insights)}"


@tool
def write_report() -> str:
    """
    Compile everything into a final Markdown report and PDF.
    """
    logger.info("Tool: write_report")
    if shared_state["data"] is None:
        return "Error: No data. Cannot generate report."
    agent = ReportWriterAgent(LLM_CONFIG)
    report = agent.compile(
        question=shared_state["question"],
        intent=shared_state["intent"],
        analysis=shared_state["analysis"],
        insights=shared_state["insights"],
        charts=shared_state["charts"],
        sql_query=shared_state["sql"]
    )
    shared_state["report"] = report
    preview = report.get("markdown", "")[:500]
    return f"Report ready. PDF available. Preview:\n{preview}"


# List of all tools for the supervisor to bind
all_bi_tools = [
    parse_intent,
    generate_sql,
    execute_sql,
    analyze_data,
    visualize,
    generate_insights,
    write_report
]