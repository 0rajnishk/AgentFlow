# intent_classifier_agent.py
"""
Intent Classifier Agent with real Gemini-based classification logic
Uses chat protocols for communication between agents
Handles queries from FastAPI/file and from any agent (including unknown/other agents).
"""

import os
import re
import json
import logging
from datetime import datetime
from uuid import uuid4
from typing import Tuple, Dict

import google.generativeai as genai
from uagents import Agent, Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)

# ---------- AGENT SETUP ----------
intent_classifier = Agent(
    name="intent_classifier", 
    mailbox=True, 
    port=9000, 
    # endpoint=["http://localhost:9000/submit"]
)

# Replace these with the actual addresses printed by each worker agent at startup
SQL_AGENT_ADDR = "agent1qdqugwl544acynqy9adencrw9kd5hs6jlu6cdmkzhqhzueeflktcgv4w3y5"
DOC_AGENT_ADDR = "agent1q2tpmsy506wtsdn0j7823s2vdm7f50l485azzc6z8lh2zk50cqwevn03e6q"
HYBRID_AGENT_ADDR = "agent1q0sh6f3n2r8azrs524chrn0e7h7p3qkm25v502jzczkrgjmtnhe972h2g64"
ERROR_AGENT_ADDR = "agent1q08hrn7j6t7ywmwdllrvl903t08sn4xd2ua2t4hy4kd6uxspqq0rgaudrpx"

WORKER_ADDRS = {SQL_AGENT_ADDR, DOC_AGENT_ADDR, HYBRID_AGENT_ADDR, ERROR_AGENT_ADDR}

chat_proto = Protocol(spec=chat_protocol_spec)

# File paths for FastAPI ↔ agents IPC
QUERY_FILE_PATH = "../query_to_agents.txt"
RESPONSE_FILE_PATH = "../response_from_agents.txt"

# In-memory mapping: request_id -> original sender (for agent-to-agent queries)
REQUEST_SENDER_MAP: Dict[str, str] = {}

# Gemini prompt template
PROMPT_TEMPLATE = """You are a precise query classifier for a medical information management system.

TASK: Classify this query into exactly ONE category:

CATEGORIES:
- document_only: background, biodata, history, doctor notes, symptoms observed, treatment plan, prescribed medications, follow-up recommendations, lifestyle adjustments, or other narrative details.
- database_only: Asks for structured data like billing, test results, admissions, discharge dates, medications, etc.(Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type	Discharge, Date, Medication, Test Results)
- hybrid: Requires both patient background and structured medical data
- unclear: Vague, incomplete, or ambiguous queries

USER: medical_info_user
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
- suggested_sources must be an array: use ["documents"] OR ["database"] OR ["documents", "database"]
- Do NOT use "both" — use ["documents", "database"] instead
- Ensure all fields are properly formatted as JSON"""

def get_gemini_response(query: str) -> str:
    """Send prompt to Gemini and return raw text response."""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    prompt = PROMPT_TEMPLATE.format(query=query)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
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
    cleaned = clean_model_response(raw)
    try:
        parsed = json.loads(cleaned)
        return parsed.get("query_type", "unclear"), parsed
    except Exception:
        logging.error("Failed to parse Gemini output: %s", cleaned)
        return "unclear", {}

# ---------- READ USER QUERY FROM FILE ----------
@intent_classifier.on_interval(period=1.0)
async def check_query_file(ctx: Context):
    if os.path.exists(QUERY_FILE_PATH):
        with open(QUERY_FILE_PATH, "r+") as f:
            query_line = f.readline().strip()
            if query_line:
                try:
                    request_id, query_text = query_line.split(":::", 1)
                    # Mark this as a file-originated request (sender=None)
                    message_text = f"{request_id}:::{query_text}"
                    msg = ChatMessage(
                        timestamp=datetime.utcnow(),
                        msg_id=uuid4(),
                        content=[TextContent(type="text", text=message_text)],
                    )
                    # Store sender as None for file-originated requests
                    REQUEST_SENDER_MAP[request_id] = None
                    await ctx.send(intent_classifier.address, msg)
                    ctx.logger.info(f"IntentClassifier: Pulled query '{query_text}' (ID: {request_id}) from file")
                except ValueError:
                    ctx.logger.error(f"Malformed line in query file: {query_line}")
                f.truncate(0)  # clear after reading

# ---------- CLASSIFY AND ROUTE MESSAGES ----------
@chat_proto.on_message(ChatMessage)
async def classify_and_route(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if not isinstance(item, TextContent):
            continue

        text = item.text

        # If this is a response from a worker agent (SQL/DOC/HYBRID/ERROR)
        if sender in WORKER_ADDRS:
            try:
                request_id, response_text = text.split(":::", 1)
            except ValueError:
                ctx.logger.error(f"Malformed response from worker: {text}")
                return

            # Check if this was a file-originated request
            orig_sender = REQUEST_SENDER_MAP.pop(request_id, None)
            if orig_sender is None:
                # Write to file for FastAPI
                with open(RESPONSE_FILE_PATH, "w") as f:
                    f.write(f"{request_id}:::{response_text}")
                ctx.logger.info(f"IntentClassifier: Wrote response for {request_id} to {RESPONSE_FILE_PATH}")
            else:
                # Forward response back to the original agent
                reply = ChatMessage(
                    timestamp=datetime.utcnow(),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=f"{request_id}:::{response_text}")],
                )
                await ctx.send(orig_sender, reply)
                ctx.logger.info(f"IntentClassifier: Routed response for {request_id} back to agent {orig_sender}")

            # Send acknowledgment to worker
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.msg_id,
            )
            await ctx.send(sender, ack)
            return

        # If this is a query from self (file) or any other agent (not a worker)
        if ":::" in text:
            try:
                request_id, query_text = text.split(":::", 1)
            except ValueError:
                ctx.logger.error(f"Malformed message format: {text}")
                return
        else:
            # Message from another agent without request_id, generate one
            request_id = str(uuid4())
            query_text = text
            if sender != intent_classifier.address:
                REQUEST_SENDER_MAP[request_id] = sender
                ctx.logger.info(f"IntentClassifier: Received query from agent {sender} (ID: {request_id}) [auto-generated]")
            # Repackage as a ChatMessage in the expected format for downstream
            msg = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=f"{request_id}:::{query_text}")],
            )

        # If sender is not self, this is a query from another agent (not a worker)
        if sender != intent_classifier.address and request_id not in REQUEST_SENDER_MAP:
            REQUEST_SENDER_MAP[request_id] = sender
            ctx.logger.info(f"IntentClassifier: Received query from agent {sender} (ID: {request_id})")

        # Run Gemini classification
        category, parsed = classify_query_with_gemini(query_text)
        ctx.logger.info(
            f"IntentClassifier: Gemini classified '{query_text}' as {category} | confidence={parsed.get('confidence')}"
        )

        # Determine destination agent
        dest_addr = ERROR_AGENT_ADDR  # default fallback

        if category == "hybrid":
            dest_addr = HYBRID_AGENT_ADDR
        elif category == "database_only":
            dest_addr = SQL_AGENT_ADDR
        elif category == "document_only":
            dest_addr = DOC_AGENT_ADDR
        else:  # unclear or any other category
            dest_addr = ERROR_AGENT_ADDR

        # Forward message to appropriate worker agent
        await ctx.send(dest_addr, msg)
        ctx.logger.info(f"IntentClassifier: Routed query (ID: {request_id}) to {category} agent ({dest_addr})")

# ---------- ACK HANDLER ----------
@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"ACK from {sender} for {msg.acknowledged_msg_id}")

# ---------- STARTUP HANDLER ----------
@intent_classifier.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"Intent Classifier started - Name: {ctx.agent.name}, Address: {ctx.agent.address}")

intent_classifier.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    intent_classifier.run()