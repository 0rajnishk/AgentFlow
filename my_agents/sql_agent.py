from uagents import Agent, Context, Model
import logging

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# This function will be called from agents.py to register handlers
def register_sql_agent_handlers(sql_agent: Agent, intent_classifier_agent: Agent):
    @sql_agent.on_message(model=AgentMessage)
    async def sql_handler(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"SQLAgent: Received query: {msg.query} (Request ID: {msg.request_id})")
        # --- REAL SQL LOGIC GOES HERE ---
        # For MVP, it's a simple string response
        response_msg = f"I am sqlAgent: I handle SQL queries. Your query: '{msg.query}'."
        # Send the response back to the intent_classifier to relay to FastAPI
        await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
        logging.info(f"SQLAgent: Sent response for {msg.request_id} back to Intent Classifier.")