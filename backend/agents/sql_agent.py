"""
SQL Generator Agent
────────────────────
Takes a structured intent and generates an optimized, safe PostgreSQL query.
Uses the schema description as context so the LLM always knows the tables.
"""

import json
import re
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage
from backend.db.connection import get_schema_description
from backend.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a PostgreSQL expert working with a sales database.
Return ONLY the SQL query — no explanation, no markdown, no commentary.

{schema}

== FEW-SHOT EXAMPLES (learn the correct patterns from these) ==

-- Q: "Show revenue by region"
SELECT
    r.name AS region,
    SUM(oi.quantity * oi.unit_price) AS revenue,
    COUNT(DISTINCT o.id) AS order_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN regions r ON r.id = o.region_id
WHERE o.status = 'completed'
GROUP BY r.name
ORDER BY revenue DESC
LIMIT 500;

-- Q: "Revenue trends by region for Q4 2025"
SELECT
    r.name AS region,
    DATE_TRUNC('month', o.order_date) AS period,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN regions r ON r.id = o.region_id
WHERE o.status = 'completed'
  AND o.order_date BETWEEN '2025-10-01' AND '2025-12-31'
GROUP BY r.name, period
ORDER BY period, revenue DESC
LIMIT 500;

-- Q: "Which products are underperforming?"
SELECT
    p.name AS product_name,
    p.category,
    SUM(oi.quantity * oi.unit_price) AS revenue,
    SUM(oi.quantity) AS units_sold,
    AVG(oi.quantity * oi.unit_price) AS avg_order_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE o.status = 'completed'
GROUP BY p.name, p.category
ORDER BY revenue ASC
LIMIT 500;

-- Q: "Top customers by revenue"
SELECT
    c.name AS customer_name,
    c.segment,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    COUNT(DISTINCT o.id) AS order_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN customers c ON c.id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.name, c.segment
ORDER BY total_revenue DESC
LIMIT 500;

-- Q: "Monthly profit margin trends"
SELECT
    DATE_TRUNC('month', o.order_date) AS period,
    SUM(oi.quantity * oi.unit_price) AS revenue,
    SUM(oi.quantity * (oi.unit_price - p.cost_price)) AS profit,
    ROUND(
        SUM(oi.quantity * (oi.unit_price - p.cost_price)) /
        NULLIF(SUM(oi.quantity * oi.unit_price), 0) * 100
    , 2) AS profit_margin_pct
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE o.status = 'completed'
GROUP BY period
ORDER BY period
LIMIT 500;

== RULES (follow always) ==
- revenue = oi.quantity * oi.unit_price (NEVER use a column called "revenue" directly)
- profit = oi.quantity * (oi.unit_price - p.cost_price)
- Always JOIN through orders → order_items → products/regions/customers
- Always filter WHERE o.status = 'completed'
- Always GROUP BY all non-aggregated columns
- For time filters: use o.order_date BETWEEN dates, not subqueries
- Keep queries flat and simple — avoid nested CTEs unless truly necessary
- LIMIT 500 always
"""


class SQLGeneratorAgent:
    """Generates optimized SQL from structured query intents."""

    def __init__(self, llm_config: dict):
        self.llm = self._build_llm(llm_config)
        self.schema = get_schema_description()
        logger.info("SQLGeneratorAgent ready")

    def _build_llm(self, cfg: dict):
        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0,
        )

    def generate(self, intent: dict) -> str:
        """Generate SQL for a given query intent."""
        logger.info("Generating SQL", analysis_type=intent.get("analysis_type"))

        system = SYSTEM_PROMPT.format(schema=self.schema)
        user_prompt = f"""
Generate a SQL query for this business intent:

{json.dumps(intent, indent=2)}

The query should answer: {intent.get('intent_summary', 'N/A')}
"""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            sql = self._clean_sql(raw)
            logger.info("SQL generated", length=len(sql))
            return sql
        except Exception as exc:
            logger.error("SQL generation failed", error=str(exc))
            return self._fallback_sql(intent)

    def _clean_sql(self, text: str) -> str:
        """Remove markdown fences and extra whitespace."""
        cleaned = re.sub(r"```(?:sql)?|```", "", text).strip()
        # Ensure it ends with semicolon
        if not cleaned.rstrip().endswith(";"):
            cleaned = cleaned.rstrip() + ";"
        return cleaned

    def _fallback_sql(self, intent: dict) -> str:
        """Safe fallback query when generation fails."""
        time_filter = ""
        tr = intent.get("time_range", {})
        if tr.get("start") and tr.get("end"):
            time_filter = f"AND o.order_date BETWEEN '{tr['start']}' AND '{tr['end']}'"

        return f"""
SELECT
    r.name AS region,
    DATE_TRUNC('month', o.order_date) AS period,
    SUM(oi.quantity * oi.unit_price) AS revenue,
    COUNT(DISTINCT o.id) AS order_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN regions r ON r.id = o.region_id
WHERE o.status = 'completed'
{time_filter}
GROUP BY r.name, period
ORDER BY period, revenue DESC
LIMIT 500;
"""