# 🧠 AI-Powered Business Intelligence Analyst
### Dynamic Multi-Agent Pipeline with LangGraph

> *Ask any business question in plain English → Get a complete analytical report in under 60 seconds*

---

## 🎯 The Problem We Solved

```
Traditional BI Workflow:
Business Question → Developer (SQL) → Analyst (Excel) → Report → Days Later

Our Solution:
Business Question → AI Pipeline → Complete Report → 60 Seconds
```

In every company, data exists but remains locked behind technical barriers. A sales manager who wants to know *"Which regions are underperforming this quarter?"* has to wait days for a technical team to query, analyze, and report — if they get an answer at all.

**We eliminated that barrier entirely.**

---

## ⚡ Live Demo Questions

Try these in the app:

| Question | What You'll See |
|---|---|
| `Show me revenue trends by region for Q4 2025` | Regional trend charts + growth rates |
| `Which products are underperforming this quarter?` | Ranked bar chart + anomaly highlights |
| `Compare sales across customer segments year over year` | Grouped comparison + YoY growth |
| `Top 5 customers by total revenue in 2025` | Ranking chart + pie share |
| `Show monthly profit margin trends for Electronics` | Trend line + 3-month forecast |
| `Detect any revenue anomalies in 2025` | Z-score anomaly detection chart |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUESTION                         │
│         "Which regions have the highest growth?"         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 LANGGRAPH ENGINE                         │
│         Dynamic routing between agents                   │
└─────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Agent 1 │  │ Agent 2 │  │ Agent 3 │
   │  Query  │→ │   SQL   │→ │  Data   │
   │ Parser  │  │ Writer  │  │ Analyst │
   └─────────┘  └─────────┘  └─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
               ┌─────────┐  ┌─────────┐  ┌─────────┐
               │ Agent 4 │  │ Agent 5 │  │ Agent 6 │
               │  Viz    │  │Insights │  │ Report  │
               │ Charts  │  │   LLM   │  │  PDF    │
               └─────────┘  └─────────┘  └─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │    COMPLETE REPORT        │
                    │  KPIs + Charts + PDF      │
                    └──────────────────────────┘
```

---

## 🔄 Why Dynamic? The LangGraph Difference

### Old Version (Deterministic)
```python
# Always runs ALL 6 steps — even if not needed
step1 → step2 → step3 → step4 → step5 → step6
```

### New Version (Dynamic with LangGraph)
```python
# LangGraph decides the path based on the data
def should_analyze(state):
    if state.get("error"):
        return END          # ← stops if SQL fails
    if not state.get("data"):
        return END          # ← stops if no data
    return "analyze"        # ← continues if data exists
```

**The agent decides at each step whether to continue, skip, or stop** — just like a real analyst would.

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────┐
│  INTELLIGENCE LAYER                          │
│  • Llama 3.1 via Ollama (100% FREE, LOCAL)   │
│  • LangChain — agent coordination            │
│  • LangGraph — dynamic workflow engine       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  BACKEND                                     │
│  • FastAPI — REST API                        │
│  • SQLAlchemy — database ORM                 │
│  • PostgreSQL — real relational database     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ANALYTICS                                   │
│  • Pandas + NumPy — data manipulation        │
│  • SciPy — statistical analysis             │
│  • Statsmodels — Holt-Winters forecasting    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  OUTPUT                                      │
│  • Plotly — interactive charts               │
│  • WeasyPrint — PDF generation               │
│  • Streamlit — web dashboard                 │
└─────────────────────────────────────────────┘
```

---

## 🧩 The 6 Agents Explained

### 🔍 Agent 1 — Query Understanding
Takes your plain English question and converts it to structured JSON intent.

```
Input:  "Show revenue trends by region for Q4 2025"

Output: {
  "analysis_type": "trend",
  "metrics": ["revenue"],
  "dimensions": ["region"],
  "time_range": { "period": "Q4 2025", "granularity": "month" },
  "filters": { "status": "completed" }
}
```

### 🗄️ Agent 2 — SQL Generator
Uses **Few-Shot Prompting** — gives the LLM real examples from our schema, not abstract rules. The model learns patterns and writes correct PostgreSQL for any question.

```sql
-- Generated automatically for the question above:
SELECT r.name AS region,
       DATE_TRUNC('month', o.order_date) AS period,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN regions r ON r.id = o.region_id
WHERE o.status = 'completed'
  AND o.order_date BETWEEN '2025-10-01' AND '2025-12-31'
GROUP BY r.name, period
ORDER BY period, revenue DESC;
```

**Self-Healing:** If SQL fails, the LLM sees the error and fixes itself automatically.

### 📊 Agent 3 — Data Analyst
Runs statistical analysis on the results:
- Growth rates and trend detection
- Z-score anomaly detection
- Holt-Winters exponential smoothing for forecasting
- Falls back to linear regression for short series

### 📈 Agent 4 — Visualization
Creates interactive Plotly charts — the type adapts to the analysis:

| Analysis Type | Charts Generated |
|---|---|
| Trend | Line chart + Growth rate bar |
| Ranking | Horizontal bar + Pie share |
| Comparison | Grouped bar chart |
| Anomaly | Scatter with highlighted outliers |
| Forecast | Line with dotted forecast extension |

### 💡 Agent 5 — Insight Generator
The LLM reads the statistical results and writes executive-level insights with specific numbers, typed as positive/warning/risk/opportunity.

### 📄 Agent 6 — Report Writer
Compiles everything into a formatted Markdown report + renders a professional PDF with charts, KPIs, insights, and the SQL query used.

---

## 🗃️ Database Design

Built from scratch with **synthetic but realistic data**:

```
5 Regions          × 12 Products × 20 Customers
North America        Electronics    Enterprise
Europe               Software       SMB  
Asia Pacific         Furniture      Consumer
Latin America        Services
Middle East          Accessories

2,000 Orders  →  5,014 Order Items
Date range: 2024 – 2026
```

**Realism built in:**
- Seasonal patterns (Q4 peaks in North America, summer dips in Europe)
- 2 intentionally underperforming products for realistic demo scenarios
- Price variance ±5% per order

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

# Seed database
python scripts/seed_database.py

# Terminal 1 — Backend
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
.venv/bin/streamlit run frontend/app.py
```

Open **http://localhost:8501** 🎉

---

## 🌟 What Makes This Different

| Feature | This Project | Typical BI Tool |
|---|---|---|
| **Language** | Plain English | SQL required |
| **Setup** | Ask a question | Configure dashboards |
| **Cost** | Free (local LLM) | $$$  subscriptions |
| **Privacy** | Data stays local | Cloud upload |
| **Output** | Full PDF report | Static charts |
| **AI** | Dynamic agent pipeline | None |

---

## 📁 Project Structure

```
bi-analyst-dynamic/
├── backend/
│   ├── main.py                      # FastAPI app
│   ├── agents/
│   │   ├── orchestrator_dynamic.py  # LangGraph pipeline ⭐
│   │   ├── query_agent.py           # NL → JSON intent
│   │   ├── sql_agent.py             # Intent → SQL (few-shot)
│   │   ├── analyst_agent.py         # Statistical analysis
│   │   ├── viz_agent.py             # Plotly charts
│   │   ├── insight_agent.py         # LLM insights
│   │   └── report_agent.py          # PDF generation
│   ├── db/connection.py             # PostgreSQL + schema
│   └── tools/stats_tools.py         # Z-score, Holt-Winters
├── frontend/
│   └── app.py                       # Streamlit dashboard
├── scripts/
│   └── seed_database.py             # Demo data generator
└── README.md
```

---

## 💬 One-Liner

> *"We turned days of technical work into a 60-second conversation — using a free, local AI that never sends your data anywhere."*

---

*Built with ❤️ using Python, LangGraph, Llama 3.1, FastAPI, and PostgreSQL*