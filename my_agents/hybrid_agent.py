"""
my_agents/hybrid_agent.py  (UPDATED for Hospital-Patient domain)

HybridAgent:
- retrieves semantically-similar patient documents via FAISS
- converts the user's natural-language question to SQL (Gemini)
- executes the SQL on HospitalPatientRecordsDataset.db
- asks Gemini to merge both evidence sources into one answer

Requirements
------------
pip install uagents google-generativeai langchain_google_genai langchain-community faiss-cpu
export GOOGLE_API_KEY="YOUR_KEY"
"""

# ────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────
import glob
import os
import json
import sqlite3
import logging
from typing import Any, Dict, List

import google.generativeai as genai
from uagents import Agent, Context, Model

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
GENAI_API_KEY =  os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
# ────────────────────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # SQLite – hospital records
# DB_PATH = "../data/HospitalPatientRecordsDataset.db"
# assert os.path.exists(DB_PATH), f"SQLite DB not found at {DB_PATH}"

db_files = glob.glob("./data/db/*.db")
if not db_files:
    raise FileNotFoundError("No .db files found in ./data/db")
DB_PATH = db_files[0]
assert os.path.exists(DB_PATH), f"SQLite database not found at {DB_PATH}"

logger.info(f"Using SQLite database at: {DB_PATH}")


# Vector store folder (patient-document chunks in FAISS)
VECTOR_FOLDER = "./vectorstores"        # keep your existing path if different
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Gemini
GOOGLE_API_KEY = GENAI_API_KEY
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# Embeddings + vectorstore loader
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def _load_vectorstore() -> FAISS | None:
    stores: List[FAISS] = []
    for d in os.listdir(VECTOR_FOLDER):
        p = os.path.join(VECTOR_FOLDER, d)
        if os.path.isdir(p):
            try:
                stores.append(
                    FAISS.load_local(p, embeddings, allow_dangerous_deserialization=True)
                )
            except Exception as e:
                logger.error(f"Could not load vectorstore {p}: {e}")
    if not stores:
        logger.warning("HybridAgent: No FAISS stores found.")
        return None
    base = stores[0]
    for other in stores[1:]:
        base.merge_from(other)
    logger.info("HybridAgent: Vectorstore loaded.")
    return base

VECTORSTORE = _load_vectorstore()


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

# ────────────────────────────────────────────────────────────────
# Agent message model
# ────────────────────────────────────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str

# ────────────────────────────────────────────────────────────────
# Hybrid-agent registration
# ────────────────────────────────────────────────────────────────
def register_hybrid_agent_handlers(hybrid_agent: Agent, intent_classifier_agent: Agent):
    @hybrid_agent.on_message(model=AgentMessage)
    async def hybrid_handler(ctx: Context, sender: str, msg: AgentMessage):
        logger.info(f"HybridAgent: Received → {msg.query} (ID: {msg.request_id})")

        # 1️ Document similarity search
        doc_context = ""
        if VECTORSTORE:
            docs = VECTORSTORE.similarity_search(msg.query, k=5)
            doc_context = "\n\n".join(d.page_content for d in docs)

        # 2️ SQL generation & execution
        sql_query = convert_to_sql(msg.query)
        try:
            data_rows = _execute_sql(sql_query)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            data_rows, sql_query = [], f"[ERROR] {e}"
        data_context = _prepare_data_summary(data_rows)

        # 3️ Gemini synthesis
        prompt = f"""You are a medical case analyst. Use BOTH the patient biodata excerpts and the database results to provide a comprehensive answer.

**User Question:** "{msg.query}"

**Patient Document Excerpts:**
{doc_context or "No relevant excerpts found."}

**Database Results (JSON):**
{data_context}

INSTRUCTIONS:
- Rely strictly on the information above.
- Structure the answer with clear headings and bullet points.
- End with a short conclusion or recommendation.
- If either source lacks enough detail, explicitly note the gap.

Answer:"""
        try:
            resp = GEMINI_MODEL.generate_content(prompt)
            answer = resp.text.strip()
        except Exception as e:
            logger.error(f"Gemini synthesis failed: {e}")
            answer = "HybridAgent encountered an error while composing the answer."

        # 4️⃣ Relay back to the Intent-Classifier agent
        await ctx.send(
            intent_classifier_agent.address,
            AgentMessage(query=answer, request_id=msg.request_id),
        )
        logger.info(f"HybridAgent: Response for {msg.request_id} sent.")

# ────────────────────────────────────────────────────────────────






# """
# my_agents/hybrid_agent.py
# HybridAgent: combines document similarity search + SQL generation/execution,
# then asks Gemini to synthesize a unified answer.

# Requirements
# ------------
# pip install uagents google-generativeai
# Set env var: export GOOGLE_API_KEY="YOUR_KEY"
# """

# import os
# import json
# import sqlite3
# import logging
# from typing import Any, Dict, List

# import google.generativeai as genai
# from uagents import Agent, Context, Model

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS

# # ────────────────────────────────
# # Globals
# # ────────────────────────────────
# logger = logging.getLogger(__name__)

# # SQLite
# DB_PATH = "project.db"
# assert os.path.exists(DB_PATH), f"SQLite DB not found at {DB_PATH}"

# # Vector DB
# DB_FOLDER = "vectorstores"
# os.makedirs(DB_FOLDER, exist_ok=True)

# # Gemini
# GOOGLE_API_KEY = 'AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA'
# genai.configure(api_key=GOOGLE_API_KEY)
# # GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-pro")
# GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# # Embeddings + vectorstore
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001", google_api_key=GOOGLE_API_KEY
# )

# def _load_vectorstore():
#     stores: List[FAISS] = []
#     for d in os.listdir(DB_FOLDER):
#         p = os.path.join(DB_FOLDER, d)
#         if os.path.isdir(p):
#             try:
#                 stores.append(
#                     FAISS.load_local(p, embeddings, allow_dangerous_deserialization=True)
#                 )
#             except Exception as e:
#                 logger.error(f"Could not load vectorstore {p}: {e}")
#     if not stores:
#         return None
#     base = stores[0]
#     for other in stores[1:]:
#         base.merge_from(other)
#     logger.info("HybridAgent: Vectorstore ready.")
#     return base

# VECTORSTORE = _load_vectorstore()

# # ────────────────────────────────
# # Schema helper (same as SQL agent)
# # ────────────────────────────────
# def _get_schema_info() -> str:
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     info: List[str] = []
#     cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
#     for tbl in [r["name"] for r in cur.fetchall()]:
#         info.append(f"\nTable: {tbl}")
#         cur.execute(f"PRAGMA table_info('{tbl}')")
#         for c in cur.fetchall():
#             info.append(f"  - {c['name']} ({c['type']})")
#     conn.close()
#     return "\n".join(info)

# # ────────────────────────────────
# # SQL generation helpers
# # ────────────────────────────────
# def _clean_sql_response(sql: str) -> str:
#     if sql.startswith("```sql"):
#         sql = sql[6:]
#     if sql.startswith("```"):
#         sql = sql[3:]
#     if sql.endswith("```"):
#         sql = sql[:-3]
#     sql = sql.strip()
#     if not sql.lower().startswith("select"):
#         raise Exception("Non-SELECT detected")
#     danger = ["drop", "delete", "insert", "update", "alter", "create"]
#     if any(k in sql.lower() for k in danger):
#         raise Exception("Dangerous keyword in SQL")
#     return sql

# def _generate_fallback_sql(nl: str) -> str:
#     nl = nl.lower()
#     if "product" in nl:
#         return "SELECT * FROM Products LIMIT 100;"
#     if "customer" in nl:
#         return "SELECT * FROM Customers LIMIT 100;"
#     return "SELECT * FROM Orders LIMIT 100;"

# def convert_to_sql(nl_query: str) -> str:
#     schema = _get_schema_info()
#     prompt = f"""You are an expert SQL query generator for supply chain analytics. Convert the natural language query into a precise SQLite query.

# {schema}

# Natural Language Query: "{nl_query}"

# CRITICAL: Only output SQL (no markdown, no extra text)."""
#     try:
#         resp = GEMINI_MODEL.generate_content(prompt, temperature=0.1)
#         sql = _clean_sql_response(resp.text)
#         logger.info(f"HybridAgent: SQL generated -> {sql}")
#         return sql
#     except Exception as e:
#         logger.error(f"Gemini SQL gen failed, fallback. {e}")
#         return _generate_fallback_sql(nl_query)

# def _exec_sql(sql: str) -> List[Dict[str, Any]]:
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     cur.execute(sql)
#     rows = [dict(r) for r in cur.fetchall()]
#     conn.close()
#     return rows

# # ────────────────────────────────
# # Data summary for prompt
# # ────────────────────────────────
# def _data_summary(data: List[Dict[str, Any]]) -> str:
#     if not data:
#         return "No data found"
#     if len(data) == 1 and len(data[0]) == 1:
#         k = list(data[0].keys())[0]
#         return f"Single value: {data[0][k]}"
#     sample = data if len(data) <= 5 else data[:5]
#     return json.dumps(sample, indent=2)

# # ────────────────────────────────
# # Agent message model
# # ────────────────────────────────
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # ────────────────────────────────
# # Hybrid agent registration
# # ────────────────────────────────
# def register_hybrid_agent_handlers(hybrid_agent: Agent, intent_classifier_agent: Agent):
#     @hybrid_agent.on_message(model=AgentMessage)
#     async def hybrid_handler(ctx: Context, sender: str, msg: AgentMessage):
#         logger.info(f"HybridAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

#         # 1. Document similarity search
#         doc_context = ""
#         if VECTORSTORE:
#             docs = VECTORSTORE.similarity_search(msg.query, k=5)
#             doc_context = "\n\n".join(d.page_content for d in docs)
#         else:
#             logger.warning("HybridAgent: No vectorstore loaded.")
        
#         # 2. SQL generation & execution
#         sql_query = convert_to_sql(msg.query)
#         try:
#             data = _exec_sql(sql_query)
#         except Exception as e:
#             data = []
#             logger.error(f"SQL execution failed: {e}")
        
#         data_context = _data_summary(data)

#         # 3. Gemini unified answer
#         prompt = f"""You are a supply chain analytics and policy expert at Syngenta. Use BOTH the policy excerpts and the database results to answer comprehensively.

# **User Question:** "{msg.query}"

# **Policy Document Excerpts:**
# {doc_context or "No relevant policy excerpts found."}

# **Database Results (JSON):**
# {data_context}

# INSTRUCTIONS:
# - Answer using only the info above.
# - Provide clear headings, bullets, and a brief conclusion/recommendations.
# - If either source lacks enough detail, mention that gap.

# Answer:"""

#         try:
#             resp = GEMINI_MODEL.generate_content(prompt, temperature=0.2)
#             answer = resp.text.strip()
#         except Exception as e:
#             logger.error(f"Gemini synthesis failed: {e}")
#             answer = "HybridAgent encountered an error while generating the answer."

#         # 4. Return to intent_classifier_agent
#         await ctx.send(
#             intent_classifier_agent.address,
#             AgentMessage(query=answer, request_id=msg.request_id),
#         )
#         logger.info(f"HybridAgent: Response for {msg.request_id} sent to Intent Classifier.")
# # ────────────────────────────────