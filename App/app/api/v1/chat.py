import datetime
import json
from typing import Any, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, Request
from app.core.logging import logger
from app.utils.security import verify_token
from app.api.v1.auth.models import User
from app.database import get_db
from sqlalchemy.orm import Session
import asyncio
import uuid # For generating request_id

# Import models from uagents_app
from uagents_app.common_models import AgentMessage, AgentResponse

router = APIRouter()

# Initialize the agent orchestrator (REMOVED, functionality now in UAgents)
# agent = AgentOrchestrator() # Remove this line

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.processing_queries: dict[str, bool] = {}
        self.connection_metadata: dict[str, dict] = {}
        # We might not strictly need processing_queries if we rely on the UAgent system's state

    async def connect(self, websocket: WebSocket, client_id: str, user_info: dict):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.processing_queries[client_id] = False # Keep for client-side state
        self.connection_metadata[client_id] = {
            "user_info": user_info,
            "connected_at": datetime.datetime.now().isoformat(),
            "query_count": 0
        }
        logger.info(f"WebSocket connection established for {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processing_queries:
            del self.processing_queries[client_id]
        if client_id in self.connection_metadata:
            metadata = self.connection_metadata[client_id]
            logger.info(f"WebSocket disconnected for {client_id}. Session stats: {metadata.get('query_count', 0)} queries processed")
            del self.connection_metadata[client_id]

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                message["timestamp"] = datetime.datetime.now().isoformat()
                await websocket.send_text(json.dumps(message))
                logger.debug(f"Sent message to {client_id}: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    def is_processing(self, client_id: str) -> bool:
        return self.processing_queries.get(client_id, False)

    def set_processing(self, client_id: str, status: bool):
        self.processing_queries[client_id] = status
        if status and client_id in self.connection_metadata:
            self.connection_metadata[client_id]["query_count"] += 1

    def get_active_connections_count(self) -> int:
        return len(self.active_connections)

manager = ConnectionManager()

def get_user_from_token(token: str, db: Session) -> User:
    """Get user from JWT token with enhanced validation"""
    try:
        payload = verify_token(token)
        if not payload:
            raise ValueError("Invalid token - verification failed")

        username = payload.get("sub")
        if not username:
            raise ValueError("Invalid token payload - missing username")

        user = db.query(User).filter(User.Username == username).first()
        if not user:
            raise ValueError(f"User not found: {username}")

        logger.info(f"User authenticated: {username} (Role: {user.role.RoleName if user.role else 'Unknown'})")
        return user

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise ValueError(f"Authentication failed: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db),
    request: Request = None # Access the FastAPI application instance
):
    client_id = None
    try:
        user = get_user_from_token(token, db)
        client_id = f"user_{user.UserId}"

        user_info = {
            "id": user.UserId,
            "username": user.Username,
            "role": user.role.RoleName if user.role else "Unknown"
        }

        await manager.connect(websocket, client_id, user_info)
        logger.info(f"WebSocket connected for user: {user.Username} (Total connections: {manager.get_active_connections_count()})")

        await manager.send_message(client_id, {
            "type": "connection",
            "status": "connected",
            "user": user_info,
            "server_info": {
                "version": "1.0.0",
                "capabilities": ["document_search", "database_query", "hybrid_analysis", "streaming_responses"]
            }
        })

        # Get the response_queues from the FastAPI app state
        response_queues = request.app.state.response_queues
        # Get the UAgent Bureau instance
        bureau = request.app.state.bureau # Make sure bureau is attached to app.state in main.py

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)

                message_type = message_data.get("type")
                logger.debug(f"Received message from {client_id}: {message_type}")

                if message_type == "query":
                    # Pass the bureau and response_queues
                    await handle_query_stream(client_id, message_data, user, bureau, response_queues)
                elif message_type == "stop":
                    await handle_stop(client_id)
                elif message_type == "ping":
                    await manager.send_message(client_id, {
                        "type": "pong",
                        "message": "Connection alive"
                    })
                else:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "error": f"Unknown message type: {message_type}",
                        "supported_types": ["query", "stop", "ping"]
                    })

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout for {client_id}")
                await manager.send_message(client_id, {
                    "type": "ping",
                    "message": "Connection check"
                })
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error from {client_id}: {e}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "error": "Invalid JSON format in message"
                })
            except Exception as e:
                logger.error(f"Error processing message from {client_id}: {e}")
                if "1001" in str(e):
                    logger.info(f"Client {client_id} disconnected normally")
                    break
                continue

    except ValueError as e:
        await websocket.close(code=4001, reason=str(e))
        logger.warning(f"WebSocket authentication failed: {e}")
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
            logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if client_id:
            try:
                await manager.send_message(client_id, {
                    "type": "error",
                    "error": "Internal server error",
                    "details": str(e)
                })
            except:
                pass
            manager.disconnect(client_id)

async def handle_query_stream(
    client_id: str,
    message_data: dict,
    user: User,
    bureau_instance: Any, # This will be the uagents.Bureau instance
    response_queues: Dict[str, asyncio.Queue[AgentResponse]]
):
    """Handle streaming query processing by sending to UAgents and relaying responses."""
    if manager.is_processing(client_id):
        await manager.send_message(client_id, {
            "type": "error",
            "error": "Another query is already being processed. Please wait or stop it."
        })
        return

    query_text = message_data.get("query", "").strip()
    if not query_text:
        await manager.send_message(client_id, {"type": "error", "error": "No query provided."})
        return
    if len(query_text) > 1000:
        await manager.send_message(client_id, {"type": "error", "error": "Query too long."})
        return

    manager.set_processing(client_id, True)
    query_start_time = datetime.datetime.now()
    request_id = str(uuid.uuid4()) # Unique ID for this specific query

    # Create a dedicated queue for this request_id and register it
    request_response_queue = asyncio.Queue()
    response_queues[request_id] = request_response_queue
    logger.info(f"Created response queue for Request ID: {request_id}")

    try:
        await manager.send_message(client_id, {
            "type": "processing_start",
            "query": query_text,
            "estimated_time": "variable"
        })

        user_context = {
            "user_id": user.UserId,
            "username": user.Username,
            "role_name": user.role.RoleName if user.role else "Unknown",
            "region": getattr(user, 'region', 'Global')
        }

        # Create AgentMessage and send to the intent_classifier agent's address
        # Ensure 'intent_classifier' agent is accessible through the bureau
        from uagents_app.intent_classifier_agent import intent_classifier # Import the agent instance

        await bureau_instance.send(
            intent_classifier.address, # This is the address of the intent_classifier agent
            AgentMessage(query=query_text, request_id=request_id, user_context=user_context)
        )
        logger.info(f"Sent query to intent_classifier agent via Bureau for Request ID: {request_id}")

        # Wait for responses from the agent system
        while manager.is_processing(client_id): # Continue as long as we expect responses
            try:
                # Wait for a message from the queue, with a timeout
                agent_response: AgentResponse = await asyncio.wait_for(request_response_queue.get(), timeout=120.0) # 2 min timeout
                logger.debug(f"Received agent response from queue for {request_id}: Type={agent_response.type}")

                # Translate AgentResponse to WebSocket message format
                websocket_message = {
                    "type": agent_response.type,
                    "query_type": agent_response.query_type,
                    "response": agent_response.response,
                    "request_id": agent_response.request_id,
                }

                if agent_response.type == "chunk" and agent_response.chunk:
                    websocket_message["chunk"] = agent_response.chunk
                if agent_response.type == "error" and agent_response.error:
                    websocket_message["error"] = agent_response.error
                if agent_response.classification:
                    websocket_message["classification"] = agent_response.classification
                if agent_response.document_sources:
                    websocket_message["document_sources"] = agent_response.document_sources
                if agent_response.sql_query:
                    websocket_message["sql_query"] = agent_response.sql_query
                if agent_response.data_results:
                    websocket_message["data_results"] = agent_response.data_results

                await manager.send_message(client_id, websocket_message)

                if agent_response.type == "final_answer" or agent_response.type == "error":
                    manager.set_processing(client_id, False) # Mark as done

            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for agent response for Request ID: {request_id}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "error": "Agent system timed out processing your request. Please try again.",
                    "request_id": request_id
                })
                manager.set_processing(client_id, False)
                break # Exit the response loop
            except Exception as e:
                logger.error(f"Error processing agent response for {request_id}: {e}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "error": f"An unexpected error occurred while receiving agent response: {str(e)}",
                    "request_id": request_id
                })
                manager.set_processing(client_id, False)
                break # Exit the response loop

    except Exception as e:
        logger.error(f"Error sending query to agent system for {client_id}: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "error": f"Failed to send query to agent system: {str(e)}",
            "request_id": request_id
        })
    finally:
        manager.set_processing(client_id, False)
        # Clean up the queue
        if request_id in response_queues:
            del response_queues[request_id]
            logger.info(f"Cleaned up response queue for Request ID: {request_id}")

        await manager.send_message(client_id, {
            "type": "processing_end",
            "duration_seconds": (datetime.datetime.now() - query_start_time).total_seconds(),
            "request_id": request_id
        })


async def handle_stop(client_id: str):
    """Handle stop processing request with confirmation"""
    # For now, this only stops the client-side processing flag.
    # To truly stop an agent mid-process, uAgents would need a cancellation mechanism,
    # which is more advanced.
    if manager.is_processing(client_id):
        manager.set_processing(client_id, False)
        logger.info(f"Processing stopped by user: {client_id}")
        await manager.send_message(client_id, {
            "type": "processing_stopped",
            "message": "Query processing has been stopped successfully."
        })
    else:
        await manager.send_message(client_id, {
            "type": "info",
            "message": "No active processing to stop."
        })

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.get_active_connections_count(),
        "timestamp": datetime.datetime.now().isoformat()
    }