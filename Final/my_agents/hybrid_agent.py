"""
my_agents/hybrid_agent.py
HybridAgent: combines document similarity search + SQL generation/execution,
then asks Gemini to synthesize a unified answer.

Requirements
------------
pip install uagents google-generativeai
Set env var: export GOOGLE_API_KEY="YOUR_KEY"
"""

import os
import json
import sqlite3
import logging
from typing import Any, Dict, List

import google.generativeai as genai
from uagents import Agent, Context, Model

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ────────────────────────────────
# Globals
# ────────────────────────────────
logger = logging.getLogger(__name__)

# SQLite
DB_PATH = "project.db"
assert os.path.exists(DB_PATH), f"SQLite DB not found at {DB_PATH}"

# Vector DB
DB_FOLDER = "vectorstores"
os.makedirs(DB_FOLDER, exist_ok=True)

# Gemini
GOOGLE_API_KEY = 'AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA'
genai.configure(api_key=GOOGLE_API_KEY)
# GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-pro")
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# Embeddings + vectorstore
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)

def _load_vectorstore():
    stores: List[FAISS] = []
    for d in os.listdir(DB_FOLDER):
        p = os.path.join(DB_FOLDER, d)
        if os.path.isdir(p):
            try:
                stores.append(
                    FAISS.load_local(p, embeddings, allow_dangerous_deserialization=True)
                )
            except Exception as e:
                logger.error(f"Could not load vectorstore {p}: {e}")
    if not stores:
        return None
    base = stores[0]
    for other in stores[1:]:
        base.merge_from(other)
    logger.info("HybridAgent: Vectorstore ready.")
    return base

VECTORSTORE = _load_vectorstore()

# ────────────────────────────────
# Schema helper (same as SQL agent)
# ────────────────────────────────
def _get_schema_info() -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    info: List[str] = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    for tbl in [r["name"] for r in cur.fetchall()]:
        info.append(f"\nTable: {tbl}")
        cur.execute(f"PRAGMA table_info('{tbl}')")
        for c in cur.fetchall():
            info.append(f"  - {c['name']} ({c['type']})")
    conn.close()
    return "\n".join(info)

# ────────────────────────────────
# SQL generation helpers
# ────────────────────────────────
def _clean_sql_response(sql: str) -> str:
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    if not sql.lower().startswith("select"):
        raise Exception("Non-SELECT detected")
    danger = ["drop", "delete", "insert", "update", "alter", "create"]
    if any(k in sql.lower() for k in danger):
        raise Exception("Dangerous keyword in SQL")
    return sql

def _generate_fallback_sql(nl: str) -> str:
    nl = nl.lower()
    if "product" in nl:
        return "SELECT * FROM Products LIMIT 100;"
    if "customer" in nl:
        return "SELECT * FROM Customers LIMIT 100;"
    return "SELECT * FROM Orders LIMIT 100;"

def convert_to_sql(nl_query: str) -> str:
    schema = _get_schema_info()
    prompt = f"""You are an expert SQL query generator for supply chain analytics. Convert the natural language query into a precise SQLite query.

{schema}

Natural Language Query: "{nl_query}"

CRITICAL: Only output SQL (no markdown, no extra text)."""
    try:
        resp = GEMINI_MODEL.generate_content(prompt, temperature=0.1)
        sql = _clean_sql_response(resp.text)
        logger.info(f"HybridAgent: SQL generated -> {sql}")
        return sql
    except Exception as e:
        logger.error(f"Gemini SQL gen failed, fallback. {e}")
        return _generate_fallback_sql(nl_query)

def _exec_sql(sql: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# ────────────────────────────────
# Data summary for prompt
# ────────────────────────────────
def _data_summary(data: List[Dict[str, Any]]) -> str:
    if not data:
        return "No data found"
    if len(data) == 1 and len(data[0]) == 1:
        k = list(data[0].keys())[0]
        return f"Single value: {data[0][k]}"
    sample = data if len(data) <= 5 else data[:5]
    return json.dumps(sample, indent=2)

# ────────────────────────────────
# Agent message model
# ────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str

# ────────────────────────────────
# Hybrid agent registration
# ────────────────────────────────
def register_hybrid_agent_handlers(hybrid_agent: Agent, intent_classifier_agent: Agent):
    @hybrid_agent.on_message(model=AgentMessage)
    async def hybrid_handler(ctx: Context, sender: str, msg: AgentMessage):
        logger.info(f"HybridAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

        # 1. Document similarity search
        doc_context = ""
        if VECTORSTORE:
            docs = VECTORSTORE.similarity_search(msg.query, k=5)
            doc_context = "\n\n".join(d.page_content for d in docs)
        else:
            logger.warning("HybridAgent: No vectorstore loaded.")
        
        # 2. SQL generation & execution
        sql_query = convert_to_sql(msg.query)
        try:
            data = _exec_sql(sql_query)
        except Exception as e:
            data = []
            logger.error(f"SQL execution failed: {e}")
        
        data_context = _data_summary(data)

        # 3. Gemini unified answer
        prompt = f"""You are a supply chain analytics and policy expert at Syngenta. Use BOTH the policy excerpts and the database results to answer comprehensively.

**User Question:** "{msg.query}"

**Policy Document Excerpts:**
{doc_context or "No relevant policy excerpts found."}

**Database Results (JSON):**
{data_context}

INSTRUCTIONS:
- Answer using only the info above.
- Provide clear headings, bullets, and a brief conclusion/recommendations.
- If either source lacks enough detail, mention that gap.

Answer:"""

        try:
            resp = GEMINI_MODEL.generate_content(prompt, temperature=0.2)
            answer = resp.text.strip()
        except Exception as e:
            logger.error(f"Gemini synthesis failed: {e}")
            answer = "HybridAgent encountered an error while generating the answer."

        # 4. Return to intent_classifier_agent
        await ctx.send(
            intent_classifier_agent.address,
            AgentMessage(query=answer, request_id=msg.request_id),
        )
        logger.info(f"HybridAgent: Response for {msg.request_id} sent to Intent Classifier.")
# ────────────────────────────────