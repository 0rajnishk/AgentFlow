from uagents import Agent, Context, Model
import logging

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# This function will be called from agents.py to register handlers
def register_document_agent_handlers(document_agent: Agent, intent_classifier_agent: Agent):
    @document_agent.on_message(model=AgentMessage)
    async def doc_handler(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"DocumentAgent: Received query: {msg.query} (Request ID: {msg.request_id})")
        # --- REAL DOCUMENT LOGIC GOES HERE ---
        response_msg = f"I am documentAgent: I handle document-related queries. Your query: '{msg.query}'."
        await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
        logging.info(f"DocumentAgent: Sent response for {msg.request_id} back to Intent Classifier.")