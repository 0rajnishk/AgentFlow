from uagents import Agent, Context, Model
import logging

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# This function will be called from agents.py to register handlers
def register_hybrid_agent_handlers(hybrid_agent: Agent, intent_classifier_agent: Agent):
    @hybrid_agent.on_message(model=AgentMessage)
    async def hybrid_handler(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"HybridAgent: Received query: {msg.query} (Request ID: {msg.request_id})")
        # --- REAL HYBRID LOGIC GOES HERE ---
        response_msg = f"I am hybridAgent: I handle both SQL and document queries. Your query: '{msg.query}'."
        await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
        logging.info(f"HybridAgent: Sent response for {msg.request_id} back to Intent Classifier.")