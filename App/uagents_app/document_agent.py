import logging
from typing import AsyncGenerator
from uagents import Agent, Context

from uagents_app.common_models import AgentMessage, AgentResponse
from app.llm.embeddings import load_vectorstore
from app.llm.prompts import get_gemini_response # Or use direct streaming if you prefer
from app.services.intent_classifier import QueryType
import google.generativeai as genai
from app.core.config import get_settings

documentAgent = Agent(name="documentAgent")

# Initialize Gemini model for streaming within the agent (if needed)
genai.configure(api_key=get_settings().GENAI_API_KEY)
_gemini_model = genai.GenerativeModel(get_settings().GENAI_MODEL)

# Load vector store once (can be passed or loaded here)
_vector_store = load_vectorstore()

# Function to generate streaming response from Gemini (similar to your _stream_gemini_response)
async def _stream_gemini_response_chunks(prompt: str) -> AsyncGenerator[str, None]:
    try:
        response = _gemini_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logging.error(f"DocumentAgent: Error streaming from Gemini: {e}")
        yield f"\n\n**Error generating document response:** {str(e)}\n\nPlease try again."


@documentAgent.on_message(model=AgentMessage)
async def handle_document_query(ctx: Context, sender: str, msg: AgentMessage):
    logging.info(f"DocumentAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

    await ctx.send(
        ctx.agents["intent_classifier"].address,
        AgentResponse(
            request_id=msg.request_id,
            response="Document Agent: Searching through policy documents...",
            query_type=QueryType.DOCUMENT_ONLY.value,
            type="status"
        )
    )

    try:
        if not _vector_store:
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Document store not available. Please ensure documents are uploaded and indexed.",
                    query_type=QueryType.DOCUMENT_ONLY.value,
                    type="error",
                    error="Document store unavailable"
                )
            )
            return

        results = _vector_store.similarity_search(msg.query, k=5)

        if not results:
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response=f"Document Agent: I couldn't find any policy documents relevant to your question: '{msg.query}'.",
                    query_type=QueryType.DOCUMENT_ONLY.value,
                    type="final_answer" # Treat as final if no documents found
                )
            )
            return

        document_sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response="Document Agent: Found relevant documents.",
                query_type=QueryType.DOCUMENT_ONLY.value,
                document_sources=document_sources,
                type="document_sources_update"
            )
        )

        context_str = "\n\n---\n\n".join([f"Document: {doc.metadata.get('title', 'Policy Document')}\nContent: {doc.page_content}" for doc in results])
        user_role = msg.user_context.get("role_name", "User") if msg.user_context else "User"

        prompt = f"""You are a knowledgeable supply chain policy expert at Syngenta. A {user_role} has asked a question about company policies and procedures.

**User Question:** "{msg.query}"

**Relevant Policy Documents:**
{context_str}

**Instructions:**
1. Provide a comprehensive, accurate answer based ONLY on the provided documents
2. Structure your response with clear headings and bullet points

**Response Structure:**
- Start with a direct answer to the question
- End with key takeaways or recommendations

**Important:** If the documents don't contain enough information to fully answer the question, clearly state what information is available and what might require additional consultation.

**Answer:**"""

        full_response_text = ""
        async for chunk_text in _stream_gemini_response_chunks(prompt):
            full_response_text += chunk_text
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="", # Empty response, actual chunk is in 'chunk' field
                    query_type=QueryType.DOCUMENT_ONLY.value,
                    chunk=chunk_text,
                    type="chunk"
                )
            )

        # Send final completion message (can be an empty chunk to signal end, or explicit)
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=full_response_text, # Send full response for final message
                query_type=QueryType.DOCUMENT_ONLY.value,
                success=True,
                type="final_answer"
            )
        )
        logging.info(f"DocumentAgent: Successfully processed and sent streaming response for Request ID {msg.request_id}.")

    except Exception as e:
        logging.error(f"DocumentAgent: Error processing document query for {msg.request_id}: {e}")
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=f"Document Agent: An error occurred while searching documents: {str(e)}",
                query_type=QueryType.DOCUMENT_ONLY.value,
                success=False,
                type="error",
                error=str(e)
            )
        )