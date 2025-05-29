"""
my_agents/document_agent.py
DocumentAgent: loads FAISS vector stores, performs similarity search (k=5),
then asks Gemini to answer in natural language using the retrieved context.

Requirements
------------
pip install uagents langchain-google-genai langchain-community google-generativeai faiss-cpu
Set env var:  export GOOGLE_API_KEY="YOUR_KEY"
"""

import os
import logging
from typing import List

import google.generativeai as genai
from uagents import Agent, Context, Model

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # (used when saving)
from langchain_community.document_loaders import PyPDFLoader        # (used when saving)

# ────────────────────────────────
# Globals: embeddings + vectorstore
# ────────────────────────────────
logger = logging.getLogger(__name__)

DB_FOLDER = "vectorstores"
os.makedirs(DB_FOLDER, exist_ok=True)

GENAI_API_KEY =  "AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA"
# GENAI_API_KEY =  "AIzaSyA9Gc5cT6cC8dgMYsITe-7FpgPVQuJ1bgQ"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GENAI_API_KEY
)

def load_vectorstore():
    """Load and merge all FAISS stores under DB_FOLDER."""
    stores: List[FAISS] = []
    for dir_name in os.listdir(DB_FOLDER):
        path = os.path.join(DB_FOLDER, dir_name)
        if os.path.isdir(path):
            try:
                logger.info(f"Attempting to load vectorstore from: {path}")
                store = FAISS.load_local(
                    path, embeddings, allow_dangerous_deserialization=True
                )
                stores.append(store)
                logger.info(f"Successfully loaded vectorstore from: {path}")
            except Exception as e:
                logger.error(f"Error loading store from {path}: {e}")
    if not stores:
        logger.warning("No vectorstores found in DB_FOLDER.")
        return None
    base = stores[0]
    for other in stores[1:]:
        base.merge_from(other)
        logger.info("Merged a vectorstore into the base vectorstore.")
    logger.info("All vectorstores loaded and merged successfully.")
    return base

VECTORSTORE = load_vectorstore()

# ────────────────────────────────
# Gemini model (text generation)
# ────────────────────────────────
genai.configure(api_key=GENAI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ────────────────────────────────
# Agent messaging model
# ────────────────────────────────
class AgentMessage(Model):
    query: str
    request_id: str

# ────────────────────────────────
# Document Agent handler registration
# ────────────────────────────────
def register_document_agent_handlers(document_agent: Agent, intent_classifier_agent: Agent):
    @document_agent.on_message(model=AgentMessage)
    async def doc_handler(ctx: Context, sender: str, msg: AgentMessage):
        logger.info(
            f"DocumentAgent: Received query: {msg.query} (Request ID: {msg.request_id})"
        )

        # 1. Similarity search
        if VECTORSTORE is None:
            answer_text = (
                "DocumentAgent could not answer because no policy documents are loaded."
            )
            await ctx.send(
                intent_classifier_agent.address,
                AgentMessage(query=answer_text, request_id=msg.request_id),
            )
            logger.warning("No vectorstore available — returning fallback answer.")
            return

        docs = VECTORSTORE.similarity_search(msg.query, k=5)
        context = "\n\n".join(d.page_content for d in docs)
        logger.info(f"Similarity search returned {len(docs)} chunks for context.")

        # 2. Build prompt for Gemini
        user_role = "employee"  # adjust as needed or add to AgentMessage
        prompt = f"""You are a knowledgeable supply chain policy expert at Syngenta. A {user_role} has asked a question about company policies and procedures.

**User Question:** "{msg.query}"

**Relevant Policy Documents:**
{context}

**Instructions:**
1. Provide a comprehensive, accurate answer based ONLY on the provided documents
2. Structure your response with clear headings and bullet points

**Response Structure:**
- Start with a direct answer to the question
- End with key takeaways or recommendations

**Important:** If the documents don't contain enough information to fully answer the question, clearly state what information is available and what might require additional consultation.

**Answer:**"""

        # 3. Call Gemini
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            answer_text = response.text
            # # Log the full response for debugging
            # logger.info(f"Gemini response: {answer_text}")
        except Exception as e:
            logger.exception("Gemini generation failed.")
            answer_text = (
                "DocumentAgent encountered an error while generating the answer."
            )

        # 4. Send answer back to IntentClassifier
        await ctx.send(
            intent_classifier_agent.address,
            AgentMessage(query=answer_text, request_id=msg.request_id),
        )

        logger.info(f"DocumentAgent: Sent response for {msg.request_id} back to Intent Classifier.")



# from uagents import Agent, Context, Model
# import logging

# # Define common models
# class AgentMessage(Model):
#     query: str
#     request_id: str

# # This function will be called from agents.py to register handlers
# def register_document_agent_handlers(document_agent: Agent, intent_classifier_agent: Agent):
#     @document_agent.on_message(model=AgentMessage)
#     async def doc_handler(ctx: Context, sender: str, msg: AgentMessage):
#         logging.info(f"DocumentAgent: Received query: {msg.query} (Request ID: {msg.request_id})")
#         # --- REAL DOCUMENT LOGIC GOES HERE ---
#         response_msg = f"I am documentAgent: I handle document-related queries. Your query: '{msg.query}'."
#         await ctx.send(intent_classifier_agent.address, AgentMessage(query=response_msg, request_id=msg.request_id))
#         logging.info(f"DocumentAgent: Sent response for {msg.request_id} back to Intent Classifier.")