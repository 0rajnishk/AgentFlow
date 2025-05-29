"""
intent_classifier_agent.py
Full implementation with Gemini-based intent classification

Requirements:
- uagents >= 0.3.1
- google-generativeai  (pip install google-generativeai)
- A valid Google AI API key exported:  export GOOGLE_API_KEY="YOUR_KEY"
"""

import os
import re
import json
import logging
from typing import Tuple

import google.generativeai as genai
from uagents import Agent, Context, Model


# ────────────────────────────────
# Common message model
# ────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str


# ────────────────────────────────
# File paths for FastAPI ↔ agents IPC
# ────────────────────────────────
QUERY_FILE_PATH = "query_to_agents.txt"
RESPONSE_FILE_PATH = "response_from_agents.txt"


# ────────────────────────────────
# Gemini helper utilities
# ────────────────────────────────
PROMPT_TEMPLATE = """You are a precise query classifier for supply chain management systems.

TASK: Classify this query into exactly ONE category:

CATEGORIES:
- document_only: Asks about policies, procedures, definitions, compliance, or guidelines
- database_only: Asks for data, metrics, statistics, lists, counts, or performance analysis
- hybrid: Requires both policy knowledge AND data analysis
- unclear: Vague, incomplete, or ambiguous queries

USER: supply_chain_user
QUERY: "{query}"

CRITICAL: Respond with ONLY this exact JSON structure (no extra text):
{{
    "query_type": "document_only",
    "confidence": 0.95,
    "reasoning": "Brief explanation",
    "keywords": ["key", "terms"],
    "suggested_sources": ["documents"]
}}

IMPORTANT:
- suggested_sources must be an array, use ["documents"] OR ["database"] OR ["documents", "database"]
- Do NOT use "both" - use ["documents", "database"] instead
- Ensure all fields are properly formatted as JSON"""

# genai.configure(api_key="AIzaSyA9Gc5cT6cC8dgMYsITe-7FpgPVQuJ1bgQ")
genai.configure(api_key="AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA")


def get_gemini_response(query: str) -> str:
    """Send prompt to Gemini and return raw text response."""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    prompt = PROMPT_TEMPLATE.format(query=query)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:  # pragma: no cover
        logging.exception("Gemini request failed")
        return f"Error: {e}"


def clean_model_response(text: str) -> str:
    """Strip markdown fences and extract the JSON blob."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group()

    return text.strip()


def classify_query_with_gemini(query: str) -> Tuple[str, dict]:
    """
    Return the category chosen by Gemini and full parsed result.
    Fallback to 'unclear' on any error.
    """
    raw = get_gemini_response(query)
    print(f"Gemini raw response: {raw}")  # Debug output
    cleaned = clean_model_response(raw)
    print(f"Cleaned Gemini response: {cleaned}")  # Debug output
    try:
        parsed = json.loads(cleaned)
        return parsed.get("query_type", "unclear"), parsed
    except Exception:  # pragma: no cover
        logging.error("Failed to parse Gemini output: %s", cleaned)
        return "unclear", {}


# ────────────────────────────────
# Registration helper (called from agents.py)
# ────────────────────────────────
def register_intent_classifier_handlers(
    intent_classifier_agent: Agent,
    sql_agent: Agent,
    document_agent: Agent,
    hybrid_agent: Agent,
    error_handler_agent: Agent,
):
    # ── Poll FastAPI-written query file ───────────────────────────
    @intent_classifier_agent.on_interval(period=1.0)
    async def check_query_file(ctx: Context):
        if os.path.exists(QUERY_FILE_PATH):
            with open(QUERY_FILE_PATH, "r+") as f:
                query_line = f.readline().strip()
                if query_line:
                    try:
                        request_id, query_text = query_line.split(":::", 1)
                        await ctx.send(
                            intent_classifier_agent.address,
                            AgentMessage(query=query_text, request_id=request_id),
                        )
                        logging.info(
                            "IntentClassifier: Pulled query '%s' (ID: %s)",
                            query_text,
                            request_id,
                        )
                    except ValueError:
                        logging.error("Malformed line in query file: %s", query_line)
                    f.truncate(0)  # clear after reading

    # ── Main on_message handler ──────────────────────────────────
    @intent_classifier_agent.on_message(model=AgentMessage)
    async def classify(ctx: Context, sender: str, msg: AgentMessage):
        print("----------"*100)
        print(f"{msg.request_id}:::{msg.query}")
        # log these above prints
        logging.info(
            "IntentClassifier: Received message from %s: %s (Request ID: %s)",
            sender,
            msg.query,
            msg.request_id,
        )
        # If message comes back from downstream agents, write the final answer
        if sender != intent_classifier_agent.address:
            with open(RESPONSE_FILE_PATH, "w") as f:
                f.write(f"{msg.request_id}:::{msg.query}")
            logging.info(
                "IntentClassifier: Wrote response for %s to %s",
                msg.request_id,
                RESPONSE_FILE_PATH,
            )
            return

        # New query – run Gemini classification
        category, parsed = classify_query_with_gemini(msg.query)
        logging.info(
            "IntentClassifier: Gemini classified '%s' as %s | confidence=%s",
            msg.query,
            category,
            parsed.get("confidence"),
        )

        # Route to appropriate agent
        if category == "hybrid":
            await ctx.send(hybrid_agent.address, msg)
        elif category == "database_only":
            await ctx.send(sql_agent.address, msg)
        elif category == "document_only":
            await ctx.send(document_agent.address, msg)
        else:
            await ctx.send(error_handler_agent.address, msg)






# import os
# import logging
# from uagents import Agent, Context, Model

# # --- Agent Models (repeated for clarity, but ideally in a common 'models.py') ---
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # --- Agent Definition ---
# intent_classifier = Agent(name="intent_classifier")

# # --- File paths for inter-process communication ---
# QUERY_FILE_PATH = "query_to_agents.txt"
# RESPONSE_FILE_PATH = "response_from_agents.txt"

# # ----------------- INTENT CLASSIFIER HANDLERS -----------------

# @intent_classifier.on_interval(period=1.0) # Check for queries every 1 second
# async def check_query_file(ctx: Context):
#     if os.path.exists(QUERY_FILE_PATH):
#         with open(QUERY_FILE_PATH, "r+") as f:
#             query_line = f.readline().strip()
#             if query_line:
#                 try:
#                     request_id, query_text = query_line.split(":::", 1)
#                     await ctx.send(
#                         intent_classifier.address, # Send to itself to trigger its on_message handler
#                         AgentMessage(query=query_text, request_id=request_id)
#                     )
#                     logging.info(f"IntentClassifier: Read query '{query_text}' (ID: {request_id}) from file. Sent to self for classification.")
#                 except ValueError:
#                     logging.error(f"IntentClassifier: Invalid query format in file: {query_line}")
#                 f.truncate(0) # Clear the file after reading

# @intent_classifier.on_message(model=AgentMessage)
# async def classify(ctx: Context, sender: str, msg: AgentMessage):
#     # This handler acts as the main entry point for new queries AND the exit point for responses.
#     logging.info(f"IntentClassifier: Received message from {sender}: {msg.query} (Request ID: {msg.request_id})")

#     # If the message is a response coming back from another agent, write it to the response file
#     # This assumes other agents send back to intent_classifier.address as the final relay point.
#     if sender != intent_classifier.address:
#         with open(RESPONSE_FILE_PATH, "w") as f:
#             f.write(f"{msg.request_id}:::{msg.query}")
#         logging.info(f"IntentClassifier: Wrote final response for {msg.request_id} to {RESPONSE_FILE_PATH}.")
#         return # Processing of this response is complete

#     # If the message is a new query (from itself, after reading the file), classify it
#     query = msg.query.lower()
#     has_sql = "sql" in query
#     has_doc = "document" in query


#     if has_sql and has_doc:
#         await ctx.send("hybridAgent address placeholder", msg) # Replaced with actual object below
#         logging.info(f"IntentClassifier: Classified '{query}' as Hybrid. Sending to Hybrid Agent.")
#     elif has_sql:
#         await ctx.send("sqlAgent address placeholder", msg) # Replaced with actual object below
#         logging.info(f"IntentClassifier: Classified '{query}' as SQL. Sending to SQL Agent.")
#     elif has_doc:
#         await ctx.send("documentAgent address placeholder", msg) # Replaced with actual object below
#         logging.info(f"IntentClassifier: Classified '{query}' as Document. Sending to Document Agent.")
#     else:
#         await ctx.send("errorHandler address placeholder", msg) # Replaced with actual object below
#         logging.info(f"IntentClassifier: Could not classify '{query}'. Sending to Error Handler.")

# # This is cleaner.

# from uagents import Agent, Context, Model
# import logging
# import os

# # Define common models
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # Define file paths
# QUERY_FILE_PATH = "query_to_agents.txt"
# RESPONSE_FILE_PATH = "response_from_agents.txt"


# def register_intent_classifier_handlers(
#     intent_classifier_agent: Agent,
#     sql_agent: Agent,
#     document_agent: Agent,
#     hybrid_agent: Agent,
#     error_handler_agent: Agent
# ):
#     @intent_classifier_agent.on_interval(period=1.0)
#     async def check_query_file(ctx: Context):
#         if os.path.exists(QUERY_FILE_PATH):
#             with open(QUERY_FILE_PATH, "r+") as f:
#                 query_line = f.readline().strip()
#                 if query_line:
#                     try:
#                         request_id, query_text = query_line.split(":::", 1)
#                         # Send to self to trigger classification
#                         await ctx.send(
#                             intent_classifier_agent.address,
#                             AgentMessage(query=query_text, request_id=request_id)
#                         )
#                         logging.info(f"IntentClassifier: Read query '{query_text}' (ID: {request_id}) from file. Sent to self for classification.")
#                     except ValueError:
#                         logging.error(f"IntentClassifier: Invalid query format in file: {query_line}")
#                     f.truncate(0) # Clear the file after reading

#     @intent_classifier_agent.on_message(model=AgentMessage)
#     async def classify(ctx: Context, sender: str, msg: AgentMessage):
#         logging.info(f"IntentClassifier: Received message from {sender}: {msg.query} (Request ID: {msg.request_id})")

#         if sender != intent_classifier_agent.address:
#             # This is a response from another agent, write it to the response file
#             with open(RESPONSE_FILE_PATH, "w") as f:
#                 f.write(f"{msg.request_id}:::{msg.query}")
#             logging.info(f"IntentClassifier: Wrote final response for {msg.request_id} to {RESPONSE_FILE_PATH}.")
#             return

#         # This is a new query from FastAPI, classify and send to appropriate agent
#         query = msg.query.lower()
#         has_sql = "sql" in query
#         has_doc = "document" in query

#         if has_sql and has_doc:
#             await ctx.send(hybrid_agent.address, msg)
#             logging.info(f"IntentClassifier: Classified '{query}' as Hybrid. Sending to Hybrid Agent.")
#         elif has_sql:
#             await ctx.send(sql_agent.address, msg)
#             logging.info(f"IntentClassifier: Classified '{query}' as SQL. Sending to SQL Agent.")
#         elif has_doc:
#             await ctx.send(document_agent.address, msg)
#             logging.info(f"IntentClassifier: Classified '{query}' as Document. Sending to Document Agent.")
#         else:
#             await ctx.send(error_handler_agent.address, msg)
#             logging.info(f"IntentClassifier: Could not classify '{query}'. Sending to Error Handler.")