from uagents import Agent, Context, Model
import logging

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# This function will be called from agents.py to register handlers
def register_error_handler_agent_handlers(error_handler_agent: Agent, intent_classifier_agent: Agent):
    @error_handler_agent.on_message(model=AgentMessage)
    async def error_handler(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"ErrorHandler: Received query: {msg.query} (Request ID: {msg.request_id})")
        # --- REAL ERROR HANDLING LOGIC GOES HERE ---
        response_msg = f"I am errorHandler: I couldn't classify your query '{msg.query}'. Can you rephrase?"
        await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
        logging.info(f"ErrorHandler: Sent response for {msg.request_id} back to Intent Classifier.")