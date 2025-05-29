import os
import logging
from uagents import Agent, Context, Model

# --- Agent Models (repeated for clarity, but ideally in a common 'models.py') ---
class AgentMessage(Model):
    query: str
    request_id: str

# --- Agent Definition ---
intent_classifier = Agent(name="intent_classifier")

# --- File paths for inter-process communication ---
QUERY_FILE_PATH = "query_to_agents.txt"
RESPONSE_FILE_PATH = "response_from_agents.txt"

# ----------------- INTENT CLASSIFIER HANDLERS -----------------

@intent_classifier.on_interval(period=1.0) # Check for queries every 1 second
async def check_query_file(ctx: Context):
    if os.path.exists(QUERY_FILE_PATH):
        with open(QUERY_FILE_PATH, "r+") as f:
            query_line = f.readline().strip()
            if query_line:
                try:
                    request_id, query_text = query_line.split(":::", 1)
                    await ctx.send(
                        intent_classifier.address, # Send to itself to trigger its on_message handler
                        AgentMessage(query=query_text, request_id=request_id)
                    )
                    logging.info(f"IntentClassifier: Read query '{query_text}' (ID: {request_id}) from file. Sent to self for classification.")
                except ValueError:
                    logging.error(f"IntentClassifier: Invalid query format in file: {query_line}")
                f.truncate(0) # Clear the file after reading

@intent_classifier.on_message(model=AgentMessage)
async def classify(ctx: Context, sender: str, msg: AgentMessage):
    # This handler acts as the main entry point for new queries AND the exit point for responses.
    logging.info(f"IntentClassifier: Received message from {sender}: {msg.query} (Request ID: {msg.request_id})")

    # If the message is a response coming back from another agent, write it to the response file
    # This assumes other agents send back to intent_classifier.address as the final relay point.
    if sender != intent_classifier.address:
        with open(RESPONSE_FILE_PATH, "w") as f:
            f.write(f"{msg.request_id}:::{msg.query}")
        logging.info(f"IntentClassifier: Wrote final response for {msg.request_id} to {RESPONSE_FILE_PATH}.")
        return # Processing of this response is complete

    # If the message is a new query (from itself, after reading the file), classify it
    query = msg.query.lower()
    has_sql = "sql" in query
    has_doc = "document" in query

    # NOTE: These addresses (sqlAgent.address, etc.) will be resolved by the Bureau.
    # We need to make sure these agents are *added* to the Bureau in agents.py.
    # The actual agent instances will be imported in agents.py.
    # For now, we'll assume their addresses are known or can be referenced.
    # In a fully modular system, you might pass agent addresses around or use a registry.
    # For this setup, the bureau implicitly links them by adding them.

    # Placeholder for sending to other agents. The actual agent instances
    # will be imported and their addresses available in the main agents.py file.
    # For a truly isolated agent file, you'd need a discovery mechanism or explicit address passing.
    # Given the constraint, we rely on the Bureau adding all agents.

    # To ensure modularity and avoid circular imports, agent instances should be defined
    # in *this* file, and their addresses used directly.
    # However, to avoid creating new instances for each import, it's common to define
    # them in a main file and refer to them.

    # For now, let's keep the `ctx.send(otherAgent.address, msg)` as is, and ensure
    # `agents.py` (the main runner) imports *all* agent instances before adding them to the bureau.

    # Let's adjust the imports slightly within each agent module to provide agent objects.
    # To fix this, you'll pass agent objects to the classify function or access them from ctx.bureau.
    # But since you asked for "simplest", keeping the agent instances in the main agents.py
    # and importing their addresses (if available) is a valid approach.

    # A better pattern for modularity would be:
    # 1. Define agent functions/handlers in individual files.
    # 2. Define agent instances (like sqlAgent = Agent(name="sqlAgent")) in the main agents.py.
    # 3. Register handlers to these instances using @sqlAgent.on_message decorator *after* import.
    # This is more complex than just putting handlers into files.

    # Let's go with the simplest: each file defines an agent instance,
    # and the main agents.py imports these instances.

    # To make these dynamic, the other agents' addresses are needed.
    # uAgents might provide a way to look up agents by name in the Bureau.
    # For simplicity, we'll keep the `ctx.send` using the direct agent object references.
    # This means `intent_classifier_agent.py` will need to import the other agent instances.
    # This creates a slight interdependency which we should avoid for true modularity.

    # Let's stick to the current structure, but clarify how it scales:
    # Each agent.py defines its handlers. The main agents.py defines instances and adds them.
    # For inter-agent communication, the agent's name is implicitly linked to its address by Bureau.

    # A common pattern for inter-agent communication without direct import is:
    # `await ctx.send(ctx.agents["sqlAgent"].address, msg)` - but this might be `uAgents` specific.
    # Your current approach where `intent_classifier` knows `sqlAgent.address` implies `sqlAgent`
    # is available in this scope.

    # Let's assume the `intent_classifier` agent will receive the actual agent instances
    # when it's initialized, or through the Bureau. For now, the simplest is to have
    # your main agents.py import and use them.

    # THIS SECTION REMAINS AS IS, as it relies on the Agent instances being accessible
    # where this handler is defined (which is `agents.py` after import)
    # The actual "logic" can be written here.

    if has_sql and has_doc:
        await ctx.send("hybridAgent address placeholder", msg) # Replaced with actual object below
        logging.info(f"IntentClassifier: Classified '{query}' as Hybrid. Sending to Hybrid Agent.")
    elif has_sql:
        await ctx.send("sqlAgent address placeholder", msg) # Replaced with actual object below
        logging.info(f"IntentClassifier: Classified '{query}' as SQL. Sending to SQL Agent.")
    elif has_doc:
        await ctx.send("documentAgent address placeholder", msg) # Replaced with actual object below
        logging.info(f"IntentClassifier: Classified '{query}' as Document. Sending to Document Agent.")
    else:
        await ctx.send("errorHandler address placeholder", msg) # Replaced with actual object below
        logging.info(f"IntentClassifier: Could not classify '{query}'. Sending to Error Handler.")

# To make this truly modular, `intent_classifier_agent.py` should NOT know
# the *instances* of other agents directly. It should either:
# 1. Look them up by name (if `uAgents` provides a registry/lookup via `ctx`).
# 2. Be given their addresses when it's added to the Bureau or initialized.

# For the given constraints, and to keep it simple, we'll slightly break strict modularity
# by having `agents.py` (the main runner) define the agent instances and register handlers
# from these individual files. This way, `agents.py` holds the 'glue'.

# Let's keep `intent_classifier_agent.py` focused purely on its own handlers.
# The `ctx.send` calls will use the globally defined agent instances from `agents.py`.

# Correcting the `classify` function to accept agent objects from `agents.py`
# This requires a slightly different pattern for adding handlers.
# Since @agent.on_message decorators are bound at definition time,
# we need to define the handlers *in* this file, but bind them to agent instances
# that are created in `agents.py`. This is getting complicated.

# Let's use the simplest, most direct modularity that matches your existing code structure:
# Each file defines its own `Agent` instance and its handlers.
# The main `agents.py` imports these instances.

# This means `my_agents/intent_classifier_agent.py` will have to import
# `sqlAgent`, `documentAgent`, etc. if it sends to their `.address`.
# This is a circular dependency problem if those agents also need to import `intent_classifier`.

# **The best simple approach for your `uAgents` constraints:**
# 1. Define common `Model` classes in a `models.py`.
# 2. Each `agent_name_agent.py` file defines a *function* that creates and returns an `Agent` instance.
# 3. The `agents.py` (main runner) calls these functions, gets the agent instances, and adds them to the bureau.

# --- Corrected my_agents/intent_classifier_agent.py ---
# This file will define the agent and its handlers. It still needs a way to refer to other agents.
# We'll use a slightly hacky but functional way for initial modularity: pass other agent addresses
# when we *register* the handler, or rely on `ctx.bureau.agents_by_name`.
# But `ctx.bureau` is not directly accessible in `on_message` decorator scope.

# Given your constraints and the direct usage of `Agent, Bureau, Context, Model`,
# and avoiding complex new patterns, the most straightforward modularization is:
# Each file exports its agent definition and handlers.
# The `intent_classifier` will have to receive other agent addresses to send messages.

# Let's export *just the handlers* from these files and keep agent instantiation in `agents.py`.
# This is cleaner.

from uagents import Agent, Context, Model
import logging
import os

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# Define file paths
QUERY_FILE_PATH = "query_to_agents.txt"
RESPONSE_FILE_PATH = "response_from_agents.txt"

# This function will be called from agents.py to register handlers
# It takes the intent_classifier_agent instance and other agent instances as arguments
def register_intent_classifier_handlers(
    intent_classifier_agent: Agent,
    sql_agent: Agent,
    document_agent: Agent,
    hybrid_agent: Agent,
    error_handler_agent: Agent
):
    @intent_classifier_agent.on_interval(period=1.0)
    async def check_query_file(ctx: Context):
        if os.path.exists(QUERY_FILE_PATH):
            with open(QUERY_FILE_PATH, "r+") as f:
                query_line = f.readline().strip()
                if query_line:
                    try:
                        request_id, query_text = query_line.split(":::", 1)
                        # Send to self to trigger classification
                        await ctx.send(
                            intent_classifier_agent.address,
                            AgentMessage(query=query_text, request_id=request_id)
                        )
                        logging.info(f"IntentClassifier: Read query '{query_text}' (ID: {request_id}) from file. Sent to self for classification.")
                    except ValueError:
                        logging.error(f"IntentClassifier: Invalid query format in file: {query_line}")
                    f.truncate(0) # Clear the file after reading

    @intent_classifier_agent.on_message(model=AgentMessage)
    async def classify(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"IntentClassifier: Received message from {sender}: {msg.query} (Request ID: {msg.request_id})")

        if sender != intent_classifier_agent.address:
            # This is a response from another agent, write it to the response file
            with open(RESPONSE_FILE_PATH, "w") as f:
                f.write(f"{msg.request_id}:::{msg.query}")
            logging.info(f"IntentClassifier: Wrote final response for {msg.request_id} to {RESPONSE_FILE_PATH}.")
            return

        # This is a new query from FastAPI, classify and send to appropriate agent
        query = msg.query.lower()
        has_sql = "sql" in query
        has_doc = "document" in query

        if has_sql and has_doc:
            await ctx.send(hybrid_agent.address, msg)
            logging.info(f"IntentClassifier: Classified '{query}' as Hybrid. Sending to Hybrid Agent.")
        elif has_sql:
            await ctx.send(sql_agent.address, msg)
            logging.info(f"IntentClassifier: Classified '{query}' as SQL. Sending to SQL Agent.")
        elif has_doc:
            await ctx.send(document_agent.address, msg)
            logging.info(f"IntentClassifier: Classified '{query}' as Document. Sending to Document Agent.")
        else:
            await ctx.send(error_handler_agent.address, msg)
            logging.info(f"IntentClassifier: Could not classify '{query}'. Sending to Error Handler.")