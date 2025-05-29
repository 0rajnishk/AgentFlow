from uagents import Model
from typing import Dict, Any, Optional

class AgentMessage(Model):
    query: str
    request_id: str
    user_context: Optional[Dict[str, Any]] = None # Pass user context to agents

class AgentResponse(Model):
    response: str
    request_id: str
    query_type: str
    classification: Optional[Dict[str, Any]] = None
    document_sources: Optional[list] = None
    sql_query: Optional[str] = None
    data_results: Optional[list] = None
    success: bool = True
    # Add fields for streaming updates if needed
    type: str # e.g., "chunk", "status", "error", "final"
    message: Optional[str] = None # For status messages
    chunk: Optional[str] = None # For streaming text chunks
    error: Optional[str] = None # For error messages