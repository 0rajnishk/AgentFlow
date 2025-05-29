import logging
import asyncio
from uagents import Agent, Context

# Import models from common_models
from uagents_app.common_models import AgentMessage, AgentResponse

# Import external services (from your existing app/services)
from app.services.intent_classifier import IntentClassifier, QueryType

# Setup a dictionary to store response queues for each request_id
# This will be passed from the main `agents.py` to allow the intent_classifier
# to put responses into a specific queue for FastAPI.
RESPONSE_QUEUES = {}

# Agent definition
intent_classifier = Agent(name="intent_classifier")

# Initialize your existing IntentClassifier service
_intent_classifier_service = IntentClassifier()

# Access Control Configuration (same as in your original AgentOrchestrator)
ROLE_PERMISSIONS = {
    "Admin": ["all_access"],
    "User": ["document_access", "basic_database_access", "region_access:Global"],
    "Finance": ["document_access", "basic_database_access", "finance_access", "region_access:Global"],
    "Planning": ["document_access", "basic_database_access", "planning_access", "region_access:Global"],
}

def _check_access(user_role: str, user_region: str, required_permissions: list[str]) -> bool:
    """Check if the user's role and region grant the required permissions."""
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])

    if "all_access" in user_permissions:
        return True

    for req_perm in required_permissions:
        if req_perm.startswith("region_access:"):
            parts = req_perm.split(":")
            if len(parts) < 2:
                logging.error(f"Malformed region_access permission: '{req_perm}'")
                return False
            requested_region = parts[1]
            if (f"region_access:{requested_region}" not in user_permissions and
                    "region_access:Global" not in user_permissions):
                logging.warning(f"Access denied for role '{user_role}' to region '{requested_region}'")
                return False
        elif req_perm not in user_permissions:
            logging.warning(f"Access denied for role '{user_role}'. Missing permission: '{req_perm}'")
            return False
    return True


@intent_classifier.on_message(model=AgentMessage)
async def classify_and_route(ctx: Context, sender: str, msg: AgentMessage):
    logging.info(f"IntentClassifier: Received message from {sender} (Request ID: {msg.request_id})")

    # If this message is a response from another agent, relay it back to FastAPI via the queue
    if sender != intent_classifier.address:
        # Check if there's a queue registered for this request_id
        if msg.request_id in RESPONSE_QUEUES:
            response_queue = RESPONSE_QUEUES[msg.request_id]
            await response_queue.put(msg) # Put the AgentResponse (which is `msg` here) into the queue
            logging.info(f"IntentClassifier: Relayed response for Request ID {msg.request_id} to FastAPI queue.")
        else:
            logging.warning(f"IntentClassifier: No response queue found for Request ID {msg.request_id}. Response dropped.")
        return # Done with this response

    # This is a new query from FastAPI (sent to intent_classifier.address directly)
    query = msg.query.lower()
    user_context = msg.user_context or {}
    user_role = user_context.get("role_name", "User")
    user_region = user_context.get("region", "Global")

    logging.info(f"IntentClassifier: Classifying query: '{query}' (User Role: {user_role})")

    try:
        # Classify the query using your existing service
        classification = _intent_classifier_service.classify_query(query, user_context)
        query_type = classification['query_type']
        required_permissions = classification.get("required_permissions", [])

        # Send classification update to FastAPI (via intent_classifier relay)
        await ctx.send(
            intent_classifier.address, # Send to self to trigger the relay logic
            AgentResponse(
                request_id=msg.request_id,
                response=f"Classified as {query_type}.",
                query_type=query_type,
                classification=classification,
                type="classification_update"
            )
        )

        # Access Control Check
        if not _check_access(user_role, user_region, required_permissions):
            await ctx.send(
                intent_classifier.address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Access Denied: You do not have permission to view this data or perform this action.",
                    query_type=query_type,
                    type="error",
                    error="Access Denied"
                )
            )
            logging.warning(f"IntentClassifier: Access denied for '{query}' (User Role: {user_role}).")
            return

        # Route to appropriate agent
        if query_type == QueryType.DATABASE_ONLY.value:
            await ctx.send(ctx.agents["sqlAgent"].address, msg) # Use ctx.agents to get address
            logging.info(f"IntentClassifier: Routed '{query}' to SQL Agent.")
        elif query_type == QueryType.DOCUMENT_ONLY.value:
            await ctx.send(ctx.agents["documentAgent"].address, msg)
            logging.info(f"IntentClassifier: Routed '{query}' to Document Agent.")
        elif query_type == QueryType.HYBRID.value:
            await ctx.send(ctx.agents["hybridAgent"].address, msg)
            logging.info(f"IntentClassifier: Routed '{query}' to Hybrid Agent.")
        else: # QueryType.UNCLEAR.value or any other
            await ctx.send(ctx.agents["errorHandler"].address, msg)
            logging.info(f"IntentClassifier: Routed '{query}' to Error Handler.")

    except Exception as e:
        logging.error(f"IntentClassifier: Error during classification or routing: {e}")
        await ctx.send(
            intent_classifier.address,
            AgentResponse(
                request_id=msg.request_id,
                response=f"An internal error occurred during classification: {str(e)}",
                query_type=QueryType.UNCLEAR.value,
                type="error",
                error="Internal Classification Error"
            )
        )

# Function to set the response queues dictionary from the main bureau runner
def set_response_queues(queues_dict: dict):
    global RESPONSE_QUEUES
    RESPONSE_QUEUES = queues_dict