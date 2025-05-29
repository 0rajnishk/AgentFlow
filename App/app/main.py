from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging import setup_logging
from app.api.v1.document import router as document_router
from app.api.v1.chat import router as chat_router
from app.api.v1.auth.auth import router as auth_router
from app.database import create_db_and_tables
from app.api.v1.auth.models import Role
from uagents import Bureau
import asyncio
import logging
from typing import Dict, Any, Deque
from collections import deque

# Import individual agent instances and helper functions from uagents_app
from uagents_app.common_models import AgentMessage, AgentResponse
from uagents_app.intent_classifier_agent import intent_classifier, set_response_queues
from uagents_app.sql_agent import sqlAgent
from uagents_app.document_agent import documentAgent
from uagents_app.hybrid_agent import hybridAgent
from uagents_app.error_handler_agent import errorHandler

setup_logging()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(document_router, prefix="/api/v1/document", tags=["document"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])

# --- UAgents Setup ---
# Use a dictionary to hold asyncio.Queue for each request_id, allowing FastAPI to await responses
response_queues: Dict[str, asyncio.Queue[AgentResponse]] = {}

# Set the response queues dictionary in the intent classifier agent
set_response_queues(response_queues)

# Create Bureau and add agents
# We can't use the `register_*_handlers` functions directly here if agents are
# defined in their own files with `@agent.on_message` directly.
# The `intent_classifier_agent.py` was adjusted to take agent instances.
# Re-adjusting the import strategy for simplicity: Each agent file exports its *configured* agent.

# A better approach: define agents in `agents_runner.py` and import handlers.
# For simplicity with your current structure, let's keep agent instances defined in `uagents_app` files
# and import them directly. The `intent_classifier` agent needs to know the addresses
# of other agents *at runtime*. The Bureau handles this by making agents available via `ctx.agents`.

bureau = Bureau()
bureau.add(intent_classifier)
bureau.add(sqlAgent)
bureau.add(documentAgent)
bureau.add(hybridAgent)
bureau.add(errorHandler)

# --- FastAPI Startup Events ---
@app.on_event("startup")
async def startup_event():
    create_db_and_tables()
    # Add default roles (existing logic)
    from sqlalchemy.orm import Session
    from app.database import SessionLocal
    from app.api.v1.auth.models import Role
    db = SessionLocal()
    try:
        if not db.query(Role).filter(Role.RoleName == "User").first():
            db.add(Role(RoleName="User"))
            db.commit()
        if not db.query(Role).filter(Role.RoleName == "Admin").first():
            db.add(Role(RoleName="Admin"))
            db.commit()
    finally:
        db.close()

    # Start the UAgents Bureau in a separate asyncio task
    logging.info("Starting UAgents Bureau as an asyncio task...")
    asyncio.create_task(bureau.run())
    logging.info("UAgent Bureau task started.")

# Expose the response_queues to be used by chat.py
app.state.response_queues = response_queues
app.state.uagent_bureau_running = False # To track if bureau is ready

# Optional: Add a health check for the UAgents Bureau
@app.get("/uagents_health")
async def uagents_health():
    # In a real scenario, you might check internal bureau status more deeply
    return {"status": "UAgents Bureau running", "connected_agents": len(bureau._agents)} # Accessing internal for demo