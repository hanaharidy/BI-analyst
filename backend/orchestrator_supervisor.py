"""
Supervisor‑based Dynamic Orchestrator (with fallback)
──────────────────────────────────────────────────────
Uses LangGraph tool‑calling. If the LLM stops before write_report,
we force a report generation at the end.
"""

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal

from backend.tools.bi_tools import all_bi_tools, shared_state, LLM_CONFIG
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── System Prompt for the Supervisor (stronger) ─────────────────────────────

SUPERVISOR_PROMPT = """You are an AI Business Intelligence supervisor. Answer the user's question by calling tools in this **exact sequence** for any full analysis request:

1. parse_intent(question)
2. generate_sql()
3. execute_sql()
4. analyze_data()
5. visualize()
6. generate_insights()
7. write_report()   ← **MUST be the last call. Never finish without it.**

If the user asks for a *partial* result (e.g., "just show me the SQL" or "only the chart"), you may stop earlier, but otherwise complete all 7 steps.

After write_report, tell the user the report is ready.

Do not repeat tools. If a tool returns an error, retry once or explain the problem.

**You MUST call write_report for any question that implies a complete analysis.**
"""

# ── Initialize LLM with bound tools ─────────────────────────────────────────

def _build_llm():
    return ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0
    )

llm = _build_llm()
llm_with_tools = llm.bind_tools(all_bi_tools)

# ── Agent Node ───────────────────────────────────────────────────────────────

def call_supervisor(state: MessagesState):
    messages_with_system = [{"role": "system", "content": SUPERVISOR_PROMPT}] + state["messages"]
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}

# ── Conditional Edge ────────────────────────────────────────────────────────

def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ── Build Graph ─────────────────────────────────────────────────────────────

def build_supervisor_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_supervisor)
    workflow.add_node("tools", ToolNode(all_bi_tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# ── Orchestrator with Fallback ──────────────────────────────────────────────

class BISupervisorOrchestrator:
    def __init__(self, llm_config: dict = None):
        self.graph = build_supervisor_graph()
        logger.info("BISupervisorOrchestrator ready (dynamic tool‑calling mode with fallback)")

    def run(self, question: str) -> dict:
        # Reset shared state
        global shared_state
        shared_state.clear()
        shared_state.update({
            "question": "",
            "intent": {},
            "sql": "",
            "data": None,
            "analysis": {},
            "charts": [],
            "insights": [],
            "report": None,
            "error": None
        })

        # Run the supervisor graph
        initial_state = {"messages": [HumanMessage(content=question)]}
        final_state = self.graph.invoke(initial_state)

        # Get final answer from LLM
        answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                answer = msg.content
                break

        # 🔁 FALLBACK: if write_report was never called, force it
        if shared_state.get("report") is None:
            logger.warning("Supervisor did not call write_report. Forcing report generation.")
            try:
                from backend.agents.report_agent import ReportWriterAgent
                agent = ReportWriterAgent(LLM_CONFIG)
                report = agent.compile(
                    question=shared_state.get("question", question),
                    intent=shared_state.get("intent", {}),
                    analysis=shared_state.get("analysis", {}),
                    insights=shared_state.get("insights", []),
                    charts=shared_state.get("charts", []),
                    sql_query=shared_state.get("sql", "")
                )
                shared_state["report"] = report
                logger.info("Fallback report generated")
            except Exception as e:
                logger.error(f"Fallback report failed: {e}")
                shared_state["report"] = {}

        report = shared_state.get("report", {})

        # Strip binary PNG data from charts (for JSON serialization)
        clean_charts = []
        for ch in shared_state.get("charts", []):
            clean_ch = ch.copy()
            clean_ch.pop("figure_png", None)
            clean_charts.append(clean_ch)

        # Ensure kpis is always a list
        kpis = report.get("kpis")
        if kpis is None:
            kpis = []

        return {
            "status": "success" if not shared_state.get("error") else "error",
            "question": question,
            "intent": shared_state.get("intent", {}),
            "sql": shared_state.get("sql", ""),
            "data_columns": list(shared_state["data"].columns) if shared_state.get("data") is not None else [],
            "analysis": shared_state.get("analysis", {}),
            "charts": clean_charts,
            "insights": shared_state.get("insights", []),
            "report_markdown": report.get("markdown", ""),
            "kpis": kpis,
            "pdf_available": report.get("pdf_bytes") is not None,
            "_pdf_bytes": report.get("pdf_bytes"),
            "elapsed_seconds": 0,
            "steps": [],
            "error": shared_state.get("error"),
            "supervisor_answer": answer
        }