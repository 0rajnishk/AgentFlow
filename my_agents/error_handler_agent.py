from datetime import datetime
from uuid import uuid4
import logging
import os

from uagents import Agent, Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

error_handler = Agent(name="errorHandler", mailbox=True, port=9004, endpoint=["http://localhost:9004/submit"])
chat_proto = Protocol(spec=chat_protocol_spec)

@error_handler.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            try:
                request_id, user_query = item.text.split(":::", 1)
            except ValueError:
                ctx.logger.error(f"Malformed message format: {item.text}")
                continue
            ctx.logger.info(f"ErrorHandler received unclassified query from {sender}: {user_query} (Request ID: {request_id})")

            # Send acknowledgement
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)

            # Gemini prompt for fallback general knowledge answer
            prompt = f"""The following question could not be classified into any known type. Please use your own general knowledge and reasoning to answer it helpfully and clearly.

User Question: \"{user_query}\"

Answer:"""

            try:
                response = gemini_model.generate_content(prompt)
                response_msg = response.text.strip() if hasattr(response, "text") else "I'm unable to answer that right now."
            except Exception as e:
                ctx.logger.error(f"Gemini error in ErrorHandler: {e}")
                response_msg = "I couldn't process your question. Please try rephrasing it."

            reply = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=f"{request_id}:::{response_msg}")],
            )
            await ctx.send(sender, reply)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for {msg.acknowledged_msg_id}")

error_handler.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    error_handler.run()
