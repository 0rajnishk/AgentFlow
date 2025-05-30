from uagents import Agent, Context, Model
import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure Gemini API
genai.configure(api_key='AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA')
# gemini_model = genai.GenerativeModel("gemini-1.5-pro")
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# Define common models
class AgentMessage(Model):
    query: str
    request_id: str

# This function will be called from agents.py to register handlers
def register_error_handler_agent_handlers(error_handler_agent: Agent, intent_classifier_agent: Agent):
    @error_handler_agent.on_message(model=AgentMessage)
    async def error_handler(ctx: Context, sender: str, msg: AgentMessage):
        logging.info(f"ErrorHandler: Received query: {msg.query} (Request ID: {msg.request_id})")

        # Gemini prompt for fallback general knowledge answer
        prompt = f"""The following question could not be classified into any known type. Please use your own general knowledge and reasoning to answer it helpfully and clearly.

User Question: "{msg.query}"

Answer:"""

        try:
            response = gemini_model.generate_content(prompt)
            response_msg = response.text.strip() if hasattr(response, "text") else "I'm unable to answer that right now."
        except Exception as e:
            logging.error(f"Gemini error in ErrorHandler: {e}")
            response_msg = "I couldn't process your question. Please try rephrasing it."

        # Send response back to intent_classifier_agent
        await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
        logging.info(f"ErrorHandler: Sent response for {msg.request_id} back to Intent Classifier.")



# from uagents import Agent, Context, Model
# import logging

# # Define common models
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # This function will be called from agents.py to register handlers
# def register_error_handler_agent_handlers(error_handler_agent: Agent, intent_classifier_agent: Agent):
#     @error_handler_agent.on_message(model=AgentMessage)
#     async def error_handler(ctx: Context, sender: str, msg: AgentMessage):
#         logging.info(f"ErrorHandler: Received query: {msg.query} (Request ID: {msg.request_id})")
#         # --- REAL ERROR HANDLING LOGIC GOES HERE ---
#         response_msg = f"I am errorHandler: I couldn't classify your query '{msg.query}'. Can you rephrase?"
#         await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
#         logging.info(f"ErrorHandler: Sent response for {msg.request_id} back to Intent Classifier.")