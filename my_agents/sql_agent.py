"""
my_agents/sql_agent.py
SQLAgent: converts NL query → SQL with Gemini, executes on SQLite
(HospitalPatientRecordsDataset.db), then gets Gemini to phrase the answer in natural language.

Requirements
------------
pip install uagents google-generativeai
Set env var:  export GOOGLE_API_KEY="YOUR_KEY"
"""

import os
import json
import sqlite3
import logging
from typing import Any, Dict, List

import google.generativeai as genai
from uagents import Agent, Context, Model
from dotenv import load_dotenv
load_dotenv()
GENAI_API_KEY =  os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
# ────────────────────────────────
# Globals
# ────────────────────────────────
logger = logging.getLogger(__name__)

DB_PATH = "../data/HospitalPatientRecordsDataset.db"
assert os.path.exists(DB_PATH), f"SQLite database not found at {DB_PATH}"



genai.configure(api_key=GENAI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ────────────────────────────────
# Agent Message model
# ────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str


# ────────────────────────────────
# Schema helper
# ────────────────────────────────
def _get_schema_info() -> str:
    """Return schema (table, columns, sample categorical values)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    info_lines: List[str] = []

    # list tables
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row["name"] for row in cursor.fetchall()]
    for table in tables:
        info_lines.append(f"\nTable: {table}")
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            # For text columns, fetch up to 10 distinct sample values
            samples = ""
            if col_type.upper() in ("TEXT", ""):
                cursor.execute(
                    f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 10"
                )
                vals = [str(r[0]) for r in cursor.fetchall()]
                if vals:
                    samples = f" (sample values: {', '.join(vals)})"
            info_lines.append(f"  - {col_name} ({col_type}){samples}")
    conn.close()
    return "\n".join(info_lines)


# ────────────────────────────────
# SQL generation helpers
# ────────────────────────────────
def _clean_sql_response(sql_query: str) -> str:
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()

    sql_lower = sql_query.lower()
    if not sql_lower.startswith("select"):
        raise Exception("Generated query must be a SELECT statement")

    dangerous = ["drop", "delete", "insert", "update", "alter", "create"]
    for kw in dangerous:
        if kw in sql_lower:
            raise Exception(f"Dangerous keyword in query: {kw}")
    return sql_query


# def _generate_fallback_sql(nl_query: str) -> str:
#     q = nl_query.lower()
#     if "product" in q:
#         return "SELECT ProductCardId, ProductName, ProductPrice FROM Products LIMIT 100;"
#     if "customer" in q:
#         return "SELECT CustomerId, Fname, Lname, City, State, Country FROM Customers LIMIT 100;"
#     if "order" in q:
#         return "SELECT OrderId, OrderDate, OrderStatus, OrderRegion, SalesPerCustomer FROM Orders LIMIT 100;"
#     return "SELECT COUNT(*) AS total_records FROM Orders;"


# Fallback SQL tuned for the hospital-patient schema
def _generate_fallback_sql(nl_query: str) -> str:
    q = nl_query.lower()
    if "count" in q or "how many" in q:
        # quick record count
        return "SELECT COUNT(*) AS total_records FROM Patients;"
    if "billing" in q or "amount" in q or "cost" in q:
        return (
            "SELECT Name, Billing_Amount "
            "FROM Patients "
            "ORDER BY Billing_Amount DESC "
            "LIMIT 25;"
        )
    if "medication" in q:
        return (
            "SELECT Name, Medication, Medical_Condition "
            "FROM Patients "
            "WHERE Medication IS NOT NULL "
            "LIMIT 50;"
        )
    if "age" in q:
        return (
            "SELECT Name, Age, Gender, Medical_Condition "
            "FROM Patients "
            "ORDER BY Age DESC "
            "LIMIT 50;"
        )
    # safe default
    return "SELECT * FROM Patients LIMIT 10;"


def convert_to_sql(natural_language_query: str) -> str:
    schema_info = _get_schema_info()
    prompt = f"""You are an expert SQL generator for hospital patient-record analytics.
Convert the natural-language question into a VALID SQLite SELECT statement.

SCHEMA DEFINITION:
{schema_info}

QUESTION: "{natural_language_query}"

CRITICAL RULES:
1. **Output ONLY the SQL query** – no markdown, comments, or extra text.
2. The query MUST start with SELECT and reference ONLY tables/columns shown in the schema.
3. Use exact column names (case-sensitive) and correct SQLite syntax.
4. Apply filters using precise column values when provided.
5. Do NOT create/alter/drop/insert/update any table.

SQL query:"""

    try:
        response = GEMINI_MODEL.generate_content(prompt, temperature=0.1)
        if not getattr(response, "text", "").strip():
            logger.warning("Gemini returned empty response; falling back.")
            return _generate_fallback_sql(natural_language_query)
        sql_query = _clean_sql_response(response.text)
        logger.info(f"[Medical DB] Generated SQL: {sql_query}")
        return sql_query
    except Exception as e:
        logger.error(f"[Medical DB] SQL generation failed: {e}")
        return _generate_fallback_sql(natural_language_query)

# def convert_to_sql(natural_language_query: str) -> str:
#     schema_info = _get_schema_info()
#     prompt = f"""You are an expert SQL query generator for supply chain analytics. Convert the natural language query into a precise, optimized SQL query.

# {schema_info}

# Natural Language Query: "{natural_language_query}"

# CRITICAL INSTRUCTIONS:
# 1. Generate ONLY the SQL query - no explanations, no markdown formatting, no additional text.
# 2. Use exact column names as specified in the schema (case-sensitive).
# 3. Use proper SQLite syntax and functions.
# 4. IMPORTANT: Work only with the existing tables (Orders, Customers, Products) and their actual columns.
# 5. Do NOT assume additional tables or columns that don't exist in the schema.
# 6. For categorical filters, use the exact "sample values" provided in the schema.
# 7. Use appropriate JOINs when querying multiple tables.

# SQL Query:"""

#     try:
#         response = GEMINI_MODEL.generate_content(prompt, temperature=0.1)
#         if not getattr(response, "text", "").strip():
#             logger.warning("Gemini returned empty response, using fallback SQL.")
#             return _generate_fallback_sql(natural_language_query)
#         sql_query = _clean_sql_response(response.text)
#         logger.info(f"Generated SQL: {sql_query}")
#         return sql_query
#     except Exception as e:
#         logger.error(f"Gemini SQL generation failed: {e}")
#         return _generate_fallback_sql(natural_language_query)


# ────────────────────────────────
# DB execution
# ────────────────────────────────
def _execute_sql(sql_query: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    data = [dict(r) for r in rows]
    conn.close()
    return data


# ────────────────────────────────
# Answer generation
# ────────────────────────────────
def _prepare_data_summary(data: List[Dict[str, Any]]) -> str:
    if not data:
        return "No data found"

    if len(data) == 1 and len(data[0]) == 1:
        key = list(data[0].keys())[0]
        return f"The direct answer value is: {data[0][key]}"

    summary_parts = [f"Total Records: {len(data)}"]
    if len(data) <= 5:
        summary_parts.append(f"All Data:\n{json.dumps(data, indent=2)}")
    else:
        summary_parts.append(f"Sample Data (first 5 rows):\n{json.dumps(data[:5], indent=2)}")
        summary_parts.append(f"[... {len(data) - 5} more rows]")
    cols = ", ".join(data[0].keys())
    summary_parts.append(f"Columns: {cols}")
    return "\n".join(summary_parts)


def generate_answer(original_query: str, sql_query: str, data: List[Dict[str, Any]]) -> str:
    data_summary = _prepare_data_summary(data)
    prompt = f"""Based on the following data, generate a helpful and natural language answer to the original question.

Original Question: "{original_query}"
SQL Query Executed: "{sql_query}"
Data Retrieved: {data_summary}

ANSWER REQUIREMENTS:
1. Provide a concise, direct answer.
2. Integrate values from 'Data Retrieved'.
3. If one value only, embed it in a sentence.
4. If list/table, summarise clearly.
5. Do NOT include the SQL query.
6. If 'No data found', say exactly that.

Answer:"""
    try:
        response = GEMINI_MODEL.generate_content(prompt, temperature=0.2)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini answer generation failed: {e}")
        return "An error occurred while generating the answer."


# ────────────────────────────────
# Agent handler registration
# ────────────────────────────────
def register_sql_agent_handlers(sql_agent: Agent, intent_classifier_agent: Agent):
    @sql_agent.on_message(model=AgentMessage)
    async def sql_handler(ctx: Context, sender: str, msg: AgentMessage):
        logger.info(f"SQLAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

        # 1. Convert NL to SQL
        sql_query = convert_to_sql(msg.query)

        # 2. Execute SQL
        try:
            data = _execute_sql(sql_query)
        except Exception as e:
            logger.exception("SQL execution failed.")
            error_resp = f"SQLAgent error: {e}"
            await ctx.send(
                intent_classifier_agent.address,
                AgentMessage(query=error_resp, request_id=msg.request_id),
            )
            return

        # 3. Gemini natural language answer
        answer_text = generate_answer(msg.query, sql_query, data)

        # 4. Send back through intent_classifier_agent
        await ctx.send(
            intent_classifier_agent.address,
            AgentMessage(query=answer_text, request_id=msg.request_id),
        )
        logger.info(f"SQLAgent: Sent response for {msg.request_id} back to Intent Classifier.")



# from uagents import Agent, Context, Model
# import logging

# # Define common models
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # This function will be called from agents.py to register handlers
# def register_sql_agent_handlers(sql_agent: Agent, intent_classifier_agent: Agent):
#     @sql_agent.on_message(model=AgentMessage)
#     async def sql_handler(ctx: Context, sender: str, msg: AgentMessage):
#         logging.info(f"SQLAgent: Received query: {msg.query} (Request ID: {msg.request_id})")
#         # --- REAL SQL LOGIC GOES HERE ---
#         # For MVP, it's a simple string response
#         response_msg = f"I am sqlAgent: I handle SQL queries. Your query: '{msg.query}'."
#         # Send the response back to the intent_classifier to relay to FastAPI
#         await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
#         logging.info(f"SQLAgent: Sent response for {msg.request_id} back to Intent Classifier.")