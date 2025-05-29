import logging
from uagents import Agent, Bureau

# Import handler registration functions and AgentMessage model from your modular files
from my_agents.intent_classifier_agent import register_intent_classifier_handlers, AgentMessage
from my_agents.sql_agent import register_sql_agent_handlers
from my_agents.document_agent import register_document_agent_handlers
from my_agents.hybrid_agent import register_hybrid_agent_handlers
from my_agents.error_handler_agent import register_error_handler_agent_handlers

# Set up basic logging for agents
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Agent Instances (defined here, as they are part of the Bureau) ---
intent_classifier = Agent(name="intent_classifier")
sqlAgent = Agent(name="sqlAgent")
documentAgent = Agent(name="documentAgent")
hybridAgent = Agent(name="hybridAgent")
errorHandler = Agent(name="errorHandler")

# --- Register Handlers from Modular Files ---
# Pass the agent instances they need to communicate with
register_intent_classifier_handlers(
    intent_classifier, sqlAgent, documentAgent, hybridAgent, errorHandler
)
register_sql_agent_handlers(sqlAgent, intent_classifier)
register_document_agent_handlers(documentAgent, intent_classifier)
register_hybrid_agent_handlers(hybridAgent, intent_classifier)
register_error_handler_agent_handlers(errorHandler, intent_classifier)

# --- Bureau Setup ---
bureau = Bureau()
bureau.add(intent_classifier)
bureau.add(sqlAgent)
bureau.add(documentAgent)
bureau.add(hybridAgent)
bureau.add(errorHandler)

if __name__ == "__main__":
    logging.info("Starting uAgents Bureau in modular setup (separate process)...")
    bureau.run() # This will run the Bureau in its own loop