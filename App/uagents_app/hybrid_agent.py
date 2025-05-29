import logging
from uagents import Agent, Context

from uagents_app.common_models import AgentMessage, AgentResponse
from app.services.nl2sql import DatabaseManager, NLToSQLConverter, SQLAnswerGenerator
from app.llm.embeddings import load_vectorstore
from app.llm.prompts import get_gemini_response
from app.services.intent_classifier import QueryType
import google.generativeai as genai
from app.core.config import get_settings

hybridAgent = Agent(name="hybridAgent")

_db_manager = DatabaseManager()
_nl2sql_converter = NLToSQLConverter()
_sql_answer_generator = SQLAnswerGenerator()
_vector_store = load_vectorstore()
genai.configure(api_key=get_settings().GENAI_API_KEY)
_gemini_model = genai.GenerativeModel(get_settings().GENAI_MODEL)


async def _stream_gemini_response_chunks(prompt: str) -> AsyncGenerator[str, None]:
    try:
        response = _gemini_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logging.error(f"HybridAgent: Error streaming from Gemini: {e}")
        yield f"\n\n**Error generating hybrid response:** {str(e)}\n\nPlease try again."


@hybridAgent.on_message(model=AgentMessage)
async def handle_hybrid_query(ctx: Context, sender: str, msg: AgentMessage):
    logging.info(f"HybridAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

    await ctx.send(
        ctx.agents["intent_classifier"].address,
        AgentResponse(
            request_id=msg.request_id,
            response="Hybrid Agent: Initiating combined document search and database analysis...",
            query_type=QueryType.HYBRID.value,
            type="status"
        )
    )

    try:
        # --- Document part ---
        document_context = ""
        document_sources = []
        if _vector_store:
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Searching documents...",
                    query_type=QueryType.HYBRID.value,
                    type="status"
                )
            )
            doc_results = _vector_store.similarity_search(msg.query, k=3) # Fewer docs for hybrid
            if doc_results:
                document_context = "\n\n---\n\n".join([doc.page_content for doc in doc_results])
                document_sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in doc_results]
                await ctx.send(
                    ctx.agents["intent_classifier"].address,
                    AgentResponse(
                        request_id=msg.request_id,
                        response="Hybrid Agent: Found relevant documents.",
                        query_type=QueryType.HYBRID.value,
                        document_sources=document_sources,
                        type="document_sources_update"
                    )
                )
            else:
                await ctx.send(
                    ctx.agents["intent_classifier"].address,
                    AgentResponse(
                        request_id=msg.request_id,
                        response="Hybrid Agent: No relevant documents found for this part of the query.",
                        query_type=QueryType.HYBRID.value,
                        type="status"
                    )
                )
        else:
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Document store not available for hybrid query.",
                    query_type=QueryType.HYBRID.value,
                    type="status"
                )
            )


        # --- SQL part ---
        sql_query_text = None
        data_results = []
        try:
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Converting to SQL query...",
                    query_type=QueryType.HYBRID.value,
                    type="status"
                )
            )
            table_names = _db_manager.get_all_table_names()
            table_schemas = {name: _db_manager.get_table_schema(name) for name in table_names}
            sql_query_text = await _nl2sql_converter.convert_to_sql(msg.query, table_schemas, msg.user_context)

            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response=f"Hybrid Agent: Generated SQL Query:\n```sql\n{sql_query_text}\n```",
                    query_type=QueryType.HYBRID.value,
                    sql_query=sql_query_text,
                    type="status"
                )
            )

            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Executing SQL query...",
                    query_type=QueryType.HYBRID.value,
                    type="status"
                )
            )
            data_results = await _db_manager.execute_sql_query(sql_query_text)

            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Successfully fetched data from database.",
                    query_type=QueryType.HYBRID.value,
                    data_results=data_results,
                    type="status"
                )
            )
        except Exception as e_sql:
            logging.warning(f"HybridAgent: SQL part failed for {msg.request_id}: {e_sql}")
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response=f"Hybrid Agent: Could not execute SQL part: {str(e_sql)}. Will proceed with document context if available.",
                    query_type=QueryType.HYBRID.value,
                    type="status",
                    error=str(e_sql)
                )
            )

        # --- Combine and Generate Final Answer ---
        combined_context = f"User Query: {msg.query}\n\n"
        if document_context:
            combined_context += f"Relevant Documents:\n{document_context}\n\n"
        if data_results:
            combined_context += f"Database Results:\n{json.dumps(data_results, indent=2)}\n\n"
        if sql_query_text:
            combined_context += f"Generated SQL Query: {sql_query_text}\n\n"

        if not document_context and not data_results:
             await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="Hybrid Agent: Could not find relevant information from documents or database for your query.",
                    query_type=QueryType.HYBRID.value,
                    type="final_answer",
                    success=False
                )
            )
             return


        final_prompt = f"""You are an advanced Syngenta AI assistant capable of synthesizing information from both policy documents and internal databases.
The user has asked a question that requires insights from both sources.

**User Question:** "{msg.query}"

**Combined Information:**
{combined_context}

**Instructions:**
1.  **Synthesize:** Combine insights from the relevant documents and database results to provide a comprehensive answer.
2.  **Clarity:** Explain complex database results in natural language.
3.  **Completeness:** Address all aspects of the user's question, using both types of data where applicable.
4.  **Formatting:** Use markdown for clear presentation, including headings, bullet points, and tables if necessary.
5.  **Source Acknowledgment:** Briefly mention if the information came from documents or database (e.g., "According to policy document X...", "Based on sales data...").
6.  If information is limited, state what you could find.

**Answer:**"""

        full_response_text = ""
        async for chunk_text in _stream_gemini_response_chunks(final_prompt):
            full_response_text += chunk_text
            await ctx.send(
                ctx.agents["intent_classifier"].address,
                AgentResponse(
                    request_id=msg.request_id,
                    response="", # Empty response, actual chunk is in 'chunk' field
                    query_type=QueryType.HYBRID.value,
                    chunk=chunk_text,
                    type="chunk"
                )
            )

        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=full_response_text,
                query_type=QueryType.HYBRID.value,
                sql_query=sql_query_text,
                data_results=data_results,
                document_sources=document_sources,
                success=True,
                type="final_answer"
            )
        )
        logging.info(f"HybridAgent: Successfully processed and sent streaming response for Request ID {msg.request_id}.")

    except Exception as e:
        logging.error(f"HybridAgent: Critical error for {msg.request_id}: {e}")
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=f"Hybrid Agent: An unexpected error occurred during hybrid processing: {str(e)}",
                query_type=QueryType.HYBRID.value,
                success=False,
                type="error",
                error=str(e)
            )
        )