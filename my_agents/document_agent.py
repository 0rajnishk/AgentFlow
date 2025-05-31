"""
my_agents/document_agent.py
DocumentAgent: loads FAISS vector stores, performs similarity search (k=5),
then asks Gemini to answer in natural language using the retrieved context.

Requirements
------------
pip install uagents uagents-core langchain-google-genai langchain-community google-generativeai faiss-cpu
Set env var:  export GOOGLE_API_KEY="YOUR_KEY"
"""

import os
import logging
from datetime import datetime
from uuid import uuid4
from typing import List

import google.generativeai as genai
from uagents import Agent, Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # (used when saving)
from langchain_community.document_loaders import PyPDFLoader        # (used when saving)
from dotenv import load_dotenv


load_dotenv()

# ────────────────────────────────
# Globals: embeddings + vectorstore
# ────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_FOLDER = "../vectorstores"
os.makedirs(DB_FOLDER, exist_ok=True)

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

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
                logger.info(f"Attempting to load vectorstore from: {path.split('---')[0]}")
                store = FAISS.load_local(
                    path, embeddings, allow_dangerous_deserialization=True
                )
                stores.append(store)
                logger.info(f"Successfully loaded vectorstore from: {path.split('---')[0]}")
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

# ────────────────────────────────────────────────────────────────
# Agent setup
# ────────────────────────────────────────────────────────────────
document_agent = Agent(name="documentAgent", mailbox=True, port=9002, endpoint=["http://localhost:9002/submit"])
chat_proto = Protocol(spec=chat_protocol_spec)

@document_agent.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    if VECTORSTORE:
        ctx.logger.info("DocumentAgent: Vector store loaded successfully - ready to process document queries")
    else:
        ctx.logger.warning("DocumentAgent: No vector store available - document queries will be limited")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            try:
                request_id, user_query = item.text.split(":::", 1)
            except ValueError:
                ctx.logger.error(f"Malformed message format: {item.text}")
                continue
            ctx.logger.info(f"DocumentAgent: Received query from {sender}: {user_query} (Request ID: {request_id})")

            # Send acknowledgement
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)

            try:
                # 1. Check if vectorstore is available
                if VECTORSTORE is None:
                    answer_text = (
                        "DocumentAgent could not answer because no policy documents are loaded. "
                        "Please ensure the vectorstore is properly configured and contains document embeddings."
                    )
                    ctx.logger.warning("No vectorstore available — returning fallback answer.")
                else:
                    # 2. Perform similarity search
                    docs = VECTORSTORE.similarity_search(user_query, k=5)
                    context = "\n\n".join(d.page_content for d in docs)
                    ctx.logger.info(f"Similarity search returned {len(docs)} chunks for context.")

                    # 3. Create prompt for Gemini
                    prompt = f"""
**User Question:** \"{user_query}\"

**Relevant Patient Documents:**
{context}

**Instructions:**
1. Answer the question using ONLY the document content above
2. Include bullet points and headings for clarity

**Response Structure:**
- Begin with a direct answer
- Then elaborate in bullet points if needed
- End with key takeaways or further steps

**Important:** If the documents don't fully answer the question, clearly state what's known and what's missing.

**Answer:**"""

                    # 4. Call Gemini for answer generation
                    try:
                        response = GEMINI_MODEL.generate_content(prompt)
                        answer_text = response.text.strip()
                        ctx.logger.info("DocumentAgent: Successfully generated answer using document context")
                    except Exception as e:
                        ctx.logger.error(f"Gemini generation failed: {e}")
                        answer_text = (
                            "DocumentAgent encountered an error while generating the answer. "
                            "Please try rephrasing your question or check if the documents contain relevant information."
                        )

            except Exception as e:
                ctx.logger.error(f"DocumentAgent processing error: {e}")
                answer_text = f"I encountered an error while processing your document query: {str(e)}"

            # Send reply in the required format
            reply = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=f"{request_id}:::{answer_text}")],
            )
            await ctx.send(sender, reply)
            ctx.logger.info(f"DocumentAgent: Response sent to {sender}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"DocumentAgent: Received acknowledgement from {sender} for {msg.acknowledged_msg_id}")

document_agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    document_agent.run()