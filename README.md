# 🧠 AI-Powered Business Intelligence Analyst

A multi-agent AI system that transforms plain English business questions into full analytical reports — complete with SQL queries, interactive charts, statistical insights, and PDF exports.

> *"Which products are underperforming this quarter?"* → Full report in 60 seconds.

---

## 🎯 Problem Statement

Business stakeholders need quick insights from complex datasets but lack SQL or data analysis skills. Creating reports manually requires technical expertise and consumes valuable analyst time.

**Before:** Business question → wait for developer + analyst + days of work  
**After:** Business question → complete report in under 60 seconds

---

## 🏗️ Architecture

```
User Question (natural language)
         ↓
┌─────────────────────────┐
│ 1. Query Understanding  │  Parses intent, metrics, time range, dimensions
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ 2. SQL Generator        │  Creates optimized PostgreSQL queries
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ 3. Data Analyst         │  Growth rates, trends, anomaly detection, forecasting
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ 4. Visualization Agent  │  Interactive Plotly charts
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ 5. Insight Generator    │  LLM-powered executive insights
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ 6. Report Writer        │  Executive Summary + PDF export
└─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Llama 3.1 (Ollama) | Natural language understanding — runs locally, free |
| Agent Framework | LangChain | Orchestrates agents and tools |
| Backend | FastAPI | REST API between agents and frontend |
| Frontend | Streamlit | Interactive dashboard UI |
| Database | PostgreSQL | Sales data storage |
| ORM | SQLAlchemy | Safe database interactions |
| Data Analysis | Pandas + NumPy + SciPy | Aggregations, growth rates, anomaly detection |
| Forecasting | Statsmodels (Holt-Winters) | Time series forecasting |
| Charts | Plotly | Interactive visualizations |
| PDF Export | WeasyPrint | HTML → PDF report generation |
| Logging | Structlog | Structured pipeline logging |

---

## 📁 Project Structure

```
bi-analyst/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── agents/
│   │   ├── orchestrator.py     # Master pipeline coordinator
│   │   ├── query_agent.py      # Natural language → structured intent
│   │   ├── sql_agent.py        # Intent → PostgreSQL query (few-shot)
│   │   ├── analyst_agent.py    # Statistical analysis & forecasting
│   │   ├── viz_agent.py        # Chart generation
│   │   ├── insight_agent.py    # LLM-powered business insights
│   │   └── report_agent.py     # Report compilation + PDF
│   ├── db/
│   │   └── connection.py       # SQLAlchemy + schema description
│   └── tools/
│       └── stats_tools.py      # Z-score anomaly, Holt-Winters forecast
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── scripts/
│   └── seed_database.py        # Demo data generator (2000 orders)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Ollama (for local LLM)

### 1. Install System Dependencies

```bash
brew install postgresql@15
brew services start postgresql@15
```

Download Ollama from [ollama.com/download](https://ollama.com/download) then:

```bash
ollama pull llama3.1
```

### 2. Set Up Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit DATABASE_URL with your PostgreSQL username
```

### 4. Initialize Database

```bash
.venv/bin/python scripts/seed_database.py
```

### 5. Run the Application

**Terminal 1 — Backend:**
```bash
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
.venv/bin/streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501)

---

## 💡 Example Questions

```
"Show me revenue trends by region for Q4 2025"
"Which products are underperforming this quarter?"
"Compare sales across customer segments year over year"
"Top 5 customers by total revenue in 2025"
"Show monthly profit margin trends for Electronics category"
"Detect any revenue anomalies in 2025"
"What is the revenue forecast for next quarter?"
"Which regions have the highest growth rate?"
```

---

## 🔑 Key Design Decisions

**Few-Shot Prompting for SQL**  
Instead of abstract rules, the SQL agent receives real examples from our schema. The model learns patterns directly, producing more accurate queries.

**Self-Healing SQL**  
If a generated query fails, the LLM receives the error message and fixes its own SQL automatically — no manual intervention needed.

**100% Local & Free**  
Llama 3.1 runs entirely on your machine via Ollama. No API keys, no costs, no data leaving your device.

**Session State Persistence**  
Results are stored in Streamlit session state so PDF downloads never wipe the page.

**Timed Pipeline**  
Every agent step is logged with timing — you can see exactly where time is spent.

---

## 📊 Demo Database

The demo database is synthetically generated with realistic characteristics:

- **2,000 orders** across 2024–2026
- **5 regions**: North America, Europe, Asia Pacific, Latin America, Middle East
- **12 products** across 5 categories
- **20 customers** across Enterprise, SMB, and Consumer segments
- **Seasonal patterns**: Q4 peaks in North America, summer dips in Europe
- **Intentional underperformers**: 2 products with suppressed sales for realistic demo scenarios

In production, connect any PostgreSQL database by updating the schema description in `backend/db/connection.py`.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze` | Run full pipeline on a question |
| POST | `/report/pdf` | Generate and download PDF report |
| GET | `/schema` | View database schema |
| GET | `/health` | Health check |

---

## 📝 Environment Variables

```env
DATABASE_URL=postgresql://username@localhost:5432/bi_analyst
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
LOG_LEVEL=INFO
API_BASE_URL=http://localhost:8000
```

To use OpenAI instead of Ollama:
```env
USE_OPENAI=true
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```