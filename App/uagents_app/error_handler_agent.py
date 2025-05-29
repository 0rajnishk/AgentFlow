import logging
from uagents import Agent, Context

from uagents_app.common_models import AgentMessage, AgentResponse
from app.services.intent_classifier import QueryType

errorHandler = Agent(name="errorHandler")

@errorHandler.on_message(model=AgentMessage)
async def handle_error_query(ctx: Context, sender: str, msg: AgentMessage):
    logging.info(f"ErrorHandler: Received unclassified query: {msg.query} (Request ID: {msg.request_id})")

    response_msg = f"I am errorHandler: I couldn't classify your query '{msg.query}'. Can you please rephrase it or provide more details? I can help with questions about company policies, sales data, or other business-related inquiries."

    await ctx.send(
        ctx.agents["intent_classifier"].address, # Send back to classifier for relay
        AgentResponse(
            request_id=msg.request_id,
            response=response_msg,
            query_type=QueryType.UNCLEAR.value,
            success=False, # Mark as not fully successful processing of intent
            type="final_answer"
        )
    )
    logging.info(f"ErrorHandler: Sent response for Request ID {msg.request_id} back to classifier.")