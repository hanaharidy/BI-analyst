# 🧠 AI-Powered Business Intelligence Analyst

### *Dynamic Multi-Agent Pipeline with LangGraph & Llama 3.1*

> *Ask any business question in plain English → Get a complete analytical report with charts, insights, and SQL in under 2 minutes – 100% local, free, and private.*

---

## 🎯 The Problem We Solved

**Traditional BI Workflow:**  
Business Question → Developer (SQL) → Analyst (Excel) → Report → Days Later

**Our Solution:**  
Business Question → AI Agent (Supervisor + Tools) → Complete Report → Under 2 Minutes

Every company has data locked behind technical barriers. A sales manager who asks *"Which regions are underperforming this quarter?"* waits days for a technical team to query, analyze, and report.

**We eliminated that barrier entirely.**

---

## ⚡ Live Demo Questions

| Question | What You'll See |
|----------|-----------------|
| `Show me revenue trends by region for Q4 2025` | Regional line chart + growth rates |
| `Which products are underperforming this quarter?` | Ranked bar chart + anomaly highlights |
| `Compare sales across customer segments year over year` | Grouped comparison + YoY growth |
| `Top 5 customers by total revenue in 2025` | Ranking chart + pie share |
| `What is the revenue forecast for next quarter?` | Historical + forecast line chart |
| `Detect any revenue anomalies in 2025` | Z‑score anomaly detection chart |

---

## 🏗️ System Architecture

The system uses a **Supervisor Agent** (LLM) that decides which tool to call next. All tools are bound to the LLM using LangGraph's tool‑calling feature.

**Tool sequence (standard full analysis):**

1. **parse_intent** – Converts natural language to structured JSON intent.
2. **generate_sql** – Writes PostgreSQL query using few‑shot prompting.
3. **execute_sql** – Runs the query. Self‑healing: if SQL fails, LLM fixes it.
4. **analyze_data** – LLM decides which statistical analyses to run (trend, ranking, anomaly, forecast).
5. **visualize** – Creates Plotly charts. Fallbacks: scalar → KPI card, time series → line chart.
6. **generate_insights** – LLM writes executive insights with numbers and actions.
7. **write_report** – Compiles Markdown report. PDF via browser "Save as Print".

The supervisor can skip steps, retry on failure, or stop early if the user asks for a partial result (e.g., "just show me the SQL").

---

## 🔄 Why Dynamic? (LangGraph Tool‑Calling)

**Old deterministic pipeline:** always runs all steps, even if not needed.

**New dynamic supervisor:** the LLM decides the next tool based on the conversation state.

```python
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"   # execute the requested tool
    return END           # finish
```

The agent decides at each step whether to continue, skip, or stop – just like a real analyst.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Llama 3.1 via Ollama (100% local, free) |
| Agent Framework | LangGraph + LangChain (tool‑calling) |
| Backend | FastAPI |
| Database | PostgreSQL + SQLAlchemy |
| Data Analysis | Pandas, NumPy, SciPy, Statsmodels |
| Charts | Plotly |
| Frontend | Streamlit |
| PDF Export | Browser "Save as PDF" (no extra dependencies) |

---

## 🧩 Detailed Tool Descriptions

### 🔍 Tool 1 — parse_intent
Converts natural language to structured JSON.

**Input:** `"Show revenue trends by region for Q4 2025"`

**Output:**
```json
{
  "analysis_type": "trend",
  "metrics": ["revenue"],
  "dimensions": ["region"],
  "time_range": { "period": "Q4 2025", "granularity": "month" }
}
```

### 🗄️ Tool 2 — generate_sql
Uses few‑shot prompting with real examples from the database schema. The LLM learns patterns and writes correct PostgreSQL.

### 🔧 Tool 3 — execute_sql
Executes the query. If an error occurs, the LLM receives the error message and tries to fix the SQL automatically (one retry).

### 📊 Tool 4 — analyze_data
The LLM decides which statistical analyses to run based on the data shape and question. Available analyses: trend, ranking, comparison, anomaly, forecast. Uses Holt‑Winters for forecasting, Z‑score for anomaly detection.

### 📈 Tool 5 — visualize
Creates Plotly charts. Smart fallbacks ensure something is always shown:
- Single numeric value → KPI card
- Time series (date + numeric) → line chart
- Otherwise → ranking bar chart + pie

### 💡 Tool 6 — generate_insights
The LLM reads the statistical results and writes executive‑level insights. Each insight has a type (positive, warning, risk, opportunity), a title, a body with specific numbers, and a recommended action.

### 📄 Tool 7 — write_report
Compiles everything into a Markdown report. The user can save it as PDF using the browser's print dialog (Ctrl+P / Cmd+P → "Save as PDF").

---

## 🗃️ Database Design

Built with synthetic but realistic data:

- **5 regions** (North America, Europe, Asia Pacific, Latin America, Middle East)
- **12 products** across 5 categories
- **20 customers** (Enterprise, SMB, Consumer segments)
- **2,000 orders** → **5,014 order items**
- Date range: 2024 – 2026
- Seasonal patterns (Q4 peaks in North America, summer dips in Europe)
- 2 intentionally underperforming products for realistic demos

---

## 🚀 Running the Project

```bash
# Clone and setup
git clone https://github.com/hanaharidy/bi-analyst-dynamic.git
cd bi-analyst-dynamic
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama pull llama3.1
ollama serve

# Seed the database
python scripts/seed_database.py

# Terminal 1 – Backend
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Terminal 2 – Frontend
.venv/bin/streamlit run frontend/app.py
```

Open **http://localhost:8501** and ask any business question.

---

## 🌟 What Makes This Different

| Feature | This Project | Traditional BI |
|---------|--------------|----------------|
| Input | Plain English | SQL or pre‑built dashboards |
| Setup | Ask a question | Days of configuration |
| Cost | Free (local LLM) | $$$ subscriptions |
| Privacy | Data never leaves your machine | Cloud upload |
| Output | Full report + charts + PDF | Static charts |
| Decision making | Dynamic LLM supervisor | Fixed pipelines |
| Self‑healing | Yes (SQL errors fixed by LLM) | No |

---

## 📁 Project Structure

```
bi-analyst-dynamic/
├── backend/
│   ├── main.py                      # FastAPI app
│   ├── agents/
│   │   ├── query_agent.py           # NL → JSON intent
│   │   ├── sql_agent.py             # Intent → SQL (few‑shot)
│   │   ├── analyst_agent.py         # Statistical analysis (LLM‑driven)
│   │   ├── viz_agent.py             # Plotly charts (with fallbacks)
│   │   ├── insight_agent.py         # Executive insights (LLM)
│   │   └── report_agent.py          # Markdown + PDF report
│   ├── tools/
│   │   ├── stats_tools.py           # Z‑score, Holt‑Winters, etc.
│   │   └── bi_tools.py              # LangChain tools for supervisor
│   ├── orchestrator_supervisor.py   # LangGraph supervisor
│   └── db/connection.py             # PostgreSQL connection
├── frontend/
│   └── app.py                       # Streamlit dashboard
├── scripts/
│   └── seed_database.py             # Synthetic data generator
└── README.md
```

---

## 🔮 Future Improvements

- **Long‑term memory** – Remember user preferences across sessions (LangGraph `BaseStore`)
- **Multi‑turn conversations** – Follow‑up questions without re‑running the whole pipeline
- **Voice input** – Speech‑to‑text for hands‑free queries
- **More chart types** – Let the LLM decide exact chart styles and colors

---

## 💬 One‑Liner

> *"We turned days of technical work into a 2‑minute conversation – using a free, local AI that never sends your data anywhere."*

---

**Built with Python, LangGraph, Llama 3.1, FastAPI, PostgreSQL, and Streamlit.**

