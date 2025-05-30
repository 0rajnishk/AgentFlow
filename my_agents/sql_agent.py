"""
my_agents/sql_agent.py
SQLAgent: converts NL query → SQL with Gemini, executes on SQLite
(project.db), then gets Gemini to phrase the answer in natural language.

Requirements
------------
pip install uagents google-generativeai
Set env var:  export GOOGLE_API_KEY="YOUR_KEY"

This module is heavily instrumented with logging for debugging and tracing:
- Logs are emitted at key steps: environment loading, DB discovery, schema extraction, SQL generation, SQL execution, and answer generation.
- Errors and warnings are logged with context.
- Use log level DEBUG for most verbose output.
"""

import glob
import os
import json
import sqlite3
import logging
from typing import Any, Dict, List
import re
import google.generativeai as genai
from uagents import Agent, Context, Model
from dotenv import load_dotenv

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables and log the process
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    logger.critical("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
logger.debug("Loaded GOOGLE_API_KEY from environment.")

# ────────────────────────────────
# Globals
# ────────────────────────────────

# Discover database file and log the process
db_files = glob.glob("./data/db/*.db")
if not db_files:
    logger.critical("No .db files found in ./data/db")
    raise FileNotFoundError("No .db files found in ./data/db")
DB_PATH = db_files[0]
assert os.path.exists(DB_PATH), f"SQLite database not found at {DB_PATH}"

logger.info(f"Using SQLite database at: {DB_PATH}")



genai.configure(api_key=GENAI_API_KEY)
logger.debug("Configured Gemini API with provided key.")
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
logger.debug("Initialized Gemini GenerativeModel.")

# ────────────────────────────────
# Agent Message model
# ────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str


# ─── helper utilities ──────────────────────────────────────────────────
def _referenced_tables(sql: str) -> List[str]:
    """
    Extract table names appearing after FROM or JOIN (very naive).
    """
    pattern = r"\\bfrom\\s+([\\w\\\"\\.]+)|\\bjoin\\s+([\\w\\\"\\.]+)"
    refs = [grp for m in re.finditer(pattern, sql, flags=re.I) for grp in m.groups() if grp]
    return [r.strip('\"').split('.')[-1] for r in refs]

def _is_sql_valid(sql: str, allowed: List[str]) -> bool:
    """Return True iff every referenced table is in allowed list."""
    return all(t in allowed for t in _referenced_tables(sql))


# ────────────────────────────────
# Schema helper
# ────────────────────────────────
def _get_schema_info() -> str:
    """
    Return schema (table, columns, sample categorical values).
    Logs schema extraction steps and errors for debugging.
    """
    try:
        info_lines: List[str] = []
        logger.debug("Connecting to SQLite DB to extract schema info.")
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # List user-defined tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            logger.debug(f"Found tables: {tables}")

            for table in tables:
                info_lines.append(f"\nTable: {table}")
                cursor.execute(f"PRAGMA table_info('{table}')")
                columns = cursor.fetchall()
                logger.debug(f"Table '{table}' columns: {[col['name'] for col in columns]}")

                for col in columns:
                    col_name = col["name"]
                    col_type = col["type"] or "UNKNOWN"
                    samples = ""

                    # For text-like columns, fetch up to 10 distinct sample values
                    if col_type.upper() in ("TEXT", ""):
                        try:
                            cursor.execute(
                                f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 10"
                            )
                            vals = [str(r[0]) for r in cursor.fetchall()]
                            if vals:
                                samples = f" (sample values: {', '.join(vals)})"
                                logger.debug(f"Sample values for {table}.{col_name}: {vals}")
                        except sqlite3.OperationalError as e:
                            samples = " (error fetching sample values)"
                            logger.warning(f"Error fetching sample values for {table}.{col_name}: {e}")

                    info_lines.append(f"  - {col_name} ({col_type}){samples}")

        logger.info("Schema info extraction successful.")
        return "\n".join(info_lines)
    except Exception as e:
        logger.error(f"Failed to extract schema info: {e}")
        return """
Table: med
  - name (TEXT) (sample values: Bobby JacksOn, LesLie TErRy, DaNnY sMitH, andrEw waTtS, adrIENNE bEll)
  - age (INTEGER)
  - gender (TEXT) (sample values: Male, Female)
  - blood_type (TEXT) (sample values: B-, A+, A-, O+, AB+)
  - medical_condition (TEXT) (sample values: Cancer, Obesity, Diabetes)
  - date_of_admission (TIMESTAMP)
  - doctor (TEXT) (sample values: Matthew Smith, Samantha Davies, Tiffany Mitchell, Kevin Wells, Kathleen Hanna)
  - hospital (TEXT) (sample values: Sons and Miller, Kim Inc, Cook PLC, Hernandez Rogers and Vang,, White-White)
  - insurance_provider (TEXT) (sample values: Blue Cross, Medicare, Aetna)
  - billing_amount (REAL)
  - room_number (INTEGER)
  - admission_type (TEXT) (sample values: Urgent, Emergency, Elective)
  - discharge_date (TIMESTAMP)
  - medication (TEXT) (sample values: Paracetamol, Ibuprofen, Aspirin, Penicillin)
  - test_results (TEXT) (sample values: Normal, Inconclusive, Abnormal)
"""

# ────────────────────────────────
# SQL generation helpers
# ────────────────────────────────
def _clean_sql_response(sql_query: str) -> str:
    """
    Cleans Gemini's SQL output and ensures safety.
    Logs the cleaning process and any issues.
    """
    logger.debug(f"Cleaning SQL response: {sql_query!r}")
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()

    sql_lower = sql_query.lower()
    if not sql_lower.startswith("select"):
        logger.error("Generated query does not start with SELECT.")
        raise Exception("Generated query must be a SELECT statement")

    dangerous = ["drop", "delete", "insert", "update", "alter", "create"]
    for kw in dangerous:
        if kw in sql_lower:
            logger.error(f"Dangerous keyword detected in query: {kw}")
            raise Exception(f"Dangerous keyword in query: {kw}")
    logger.debug(f"Cleaned SQL query: {sql_query}")
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

# you can add your code here – Updated fallback SQL for the `med` schema
def _generate_fallback_sql(nl_query: str) -> str:
    """
    Very loose heuristics when Gemini is silent or errors out.
    Logs which fallback path is taken.
    """
    q = nl_query.lower()
    logger.warning(f"Using fallback SQL for query: {nl_query!r}")

    # simple record count
    if any(kw in q for kw in ("count", "how many", "total")):
        logger.debug("Fallback: Detected count/total query.")
        return "SELECT COUNT(*) AS total_records FROM med;"

    # billing, money, cost questions
    if any(kw in q for kw in ("billing", "amount", "cost", "expense")):
        logger.debug("Fallback: Detected billing/cost query.")
        return (
            "SELECT Name, Billing_Amount "
            "FROM med "
            "ORDER BY Billing_Amount DESC "
            "LIMIT 25;"
        )

    # medication-related queries
    if "medication" in q or "drug" in q:
        logger.debug("Fallback: Detected medication/drug query.")
        return (
            "SELECT Name, Medication, Medical_Condition "
            "FROM med "
            "WHERE Medication IS NOT NULL "
            "LIMIT 50;"
        )

    # age-based look-ups
    if "age" in q or "oldest" in q or "youngest" in q:
        logger.debug("Fallback: Detected age/oldest/youngest query.")
        return (
            "SELECT Name, Age, Gender, Medical_Condition "
            "FROM med "
            "ORDER BY Age DESC "
            "LIMIT 50;"
        )

    # recent admissions
    if "admission" in q or "admitted" in q:
        logger.debug("Fallback: Detected admission/admitted query.")
        return (
            "SELECT Name, Date_of_Admission AS Admission_Date, Admission_Type "
            "FROM med "
            "ORDER BY Date_of_Admission DESC "
            "LIMIT 25;"
        )

    # safe default
    logger.debug("Fallback: Using safe default query.")
    return "SELECT * FROM med LIMIT 10;"

def convert_to_sql(natural_language_query: str) -> str:
    """
    Convert a natural-language question into a single VALID SQLite SELECT
    statement that only touches the `med` table.

    - If Gemini hallucinates other tables, we re-prompt once with a
      hard reminder. If it still fails, fall back to heuristics.
    """
    logger.info(f"Converting NL query to SQL: {natural_language_query!r}")

    schema_info = _get_schema_info()
    base_prompt = f"""You are an expert SQL generator for hospital patient-record
analytics.  Convert the user's question into a VALID SQLite SELECT query.

SCHEMA:
{schema_info}

QUESTION: "{natural_language_query}"

RULES:
1. Output ONLY the SQL (no markdown or comments).
2. Use the exact column names shown.
3. The ONLY table is "med". No joins to other tables.
4. The query MUST start with SELECT and never modify data.

SQL query:"""

    allowed_tables = ["med"]  # <<-- the only real table in our DB

    try:
        # ── first attempt ───────────────────────────────────────────
        logger.debug("Gemini prompt (attempt 1)")
        response = GEMINI_MODEL.generate_content(base_prompt)
        sql_query = _clean_sql_response(response.text)

        # validate: check referenced tables
        if not _is_sql_valid(sql_query, allowed_tables):
            bad_tables = _referenced_tables(sql_query)
            logger.warning(f"Gemini used invalid tables {bad_tables}; retrying.")

            # ── second attempt (hard reminder) ─────────────────────
            retry_prompt = (
                base_prompt
                + "\n\nIMPORTANT: You MUST use only the table \"med\". "
                  "Do NOT invent or join other tables. Regenerate now."
            )
            response = GEMINI_MODEL.generate_content(retry_prompt)
            sql_query = _clean_sql_response(response.text)

        # final validation
        if not _is_sql_valid(sql_query, allowed_tables):
            raise ValueError("Gemini still produced invalid tables after retry.")

        logger.info(f"[Medical DB] Generated SQL: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"SQL generation failed or invalid: {e}. Falling back.")
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
    """
    Executes the given SQL query and returns the result as a list of dicts.
    Logs the query, execution, and any errors.
    """
    logger.info(f"Executing SQL query: {sql_query}")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        data = [dict(r) for r in rows]
        logger.info(f"SQL execution returned {len(data)} rows.")
        return data
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise
    finally:
        try:
            conn.close()
            logger.debug("Closed SQLite connection.")
        except Exception:
            pass

# ────────────────────────────────
# Answer generation
# ────────────────────────────────
def _prepare_data_summary(data: List[Dict[str, Any]]) -> str:
    """
    Prepares a summary of the data for answer generation.
    Logs the summary process.
    """
    logger.debug(f"Preparing data summary for {len(data)} records.")
    if not data:
        logger.debug("No data found in SQL result.")
        return "No data found"

    if len(data) == 1 and len(data[0]) == 1:
        key = list(data[0].keys())[0]
        logger.debug(f"Single value answer: {data[0][key]}")
        return f"The direct answer value is: {data[0][key]}"

    summary_parts = [f"Total Records: {len(data)}"]
    if len(data) <= 5:
        summary_parts.append(f"All Data:\n{json.dumps(data, indent=2)}")
    else:
        summary_parts.append(f"Sample Data (first 5 rows):\n{json.dumps(data[:5], indent=2)}")
        summary_parts.append(f"[... {len(data) - 5} more rows]")
    cols = ", ".join(data[0].keys())
    summary_parts.append(f"Columns: {cols}")
    logger.debug("Data summary prepared.")
    return "\n".join(summary_parts)

def generate_answer(original_query: str, sql_query: str, data: List[Dict[str, Any]]) -> str:
    """
    Generates a natural language answer using Gemini.
    Logs the prompt, Gemini response, and errors.
    """
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
        logger.debug(f"Sending answer generation prompt to Gemini: {prompt}")
        response = GEMINI_MODEL.generate_content(prompt)
        logger.info("Gemini answer generation successful.")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini answer generation failed: {e}")
        return "An error occurred while generating the answer."


# ────────────────────────────────
# Agent handler registration
# ────────────────────────────────
def register_sql_agent_handlers(sql_agent: Agent, intent_classifier_agent: Agent):
    """
    Registers the SQL agent message handler.
    Logs each step of the agent's message processing for debugging.
    """
    @sql_agent.on_message(model=AgentMessage)
    async def sql_handler(ctx: Context, sender: str, msg: AgentMessage):
        logger.info(f"SQLAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

        # 1. Convert NL to SQL
        sql_query = convert_to_sql(msg.query)
        logger.info(f"SQLAgent: Generated SQL query: {sql_query}")

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
            logger.info(f"SQLAgent: Sent error response for {msg.request_id} back to Intent Classifier.")
            return

        # 3. Gemini natural language answer
        answer_text = generate_answer(msg.query, sql_query, data)
        logger.info(f"SQLAgent: Generated answer for request {msg.request_id}.")

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
