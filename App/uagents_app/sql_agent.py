import logging
from uagents import Agent, Context

from uagents_app.common_models import AgentMessage, AgentResponse
from app.services.nl2sql import DatabaseManager, NLToSQLConverter, SQLAnswerGenerator # Your existing services
from app.services.intent_classifier import QueryType # For setting query_type

sqlAgent = Agent(name="sqlAgent")

_db_manager = DatabaseManager()
_nl2sql_converter = NLToSQLConverter()
_sql_answer_generator = SQLAnswerGenerator()

@sqlAgent.on_message(model=AgentMessage)
async def handle_sql_query(ctx: Context, sender: str, msg: AgentMessage):
    logging.info(f"SQLAgent: Received query: {msg.query} (Request ID: {msg.request_id})")

    # Send status update
    await ctx.send(
        ctx.agents["intent_classifier"].address, # Send back to classifier for relay
        AgentResponse(
            request_id=msg.request_id,
            response="SQL Agent: Analyzing your request and preparing a database query...",
            query_type=QueryType.DATABASE_ONLY.value,
            type="status"
        )
    )

    try:
        # --- Your existing NL2SQL logic ---
        table_names = _db_manager.get_all_table_names()
        table_schemas = {name: _db_manager.get_table_schema(name) for name in table_names}

        # Step 1: Convert natural language to SQL
        sql_query_text = await _nl2sql_converter.convert_to_sql(msg.query, table_schemas, msg.user_context)

        # Send SQL query as a status update
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=f"SQL Agent: Generated SQL Query:\n```sql\n{sql_query_text}\n```",
                query_type=QueryType.DATABASE_ONLY.value,
                sql_query=sql_query_text,
                type="status"
            )
        )

        # Step 2: Execute SQL query
        data_results = await _db_manager.execute_sql_query(sql_query_text)

        # Step 3: Generate natural language answer from SQL results
        final_answer = await _sql_answer_generator.generate_answer(msg.query, sql_query_text, data_results)

        # Send final response
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=final_answer,
                query_type=QueryType.DATABASE_ONLY.value,
                sql_query=sql_query_text,
                data_results=data_results,
                result_count=len(data_results) if data_results else 0,
                success=True,
                type="final_answer"
            )
        )
        logging.info(f"SQLAgent: Successfully processed and sent final response for Request ID {msg.request_id}.")

    except Exception as e:
        logging.error(f"SQLAgent: Error processing SQL query for {msg.request_id}: {e}")
        await ctx.send(
            ctx.agents["intent_classifier"].address,
            AgentResponse(
                request_id=msg.request_id,
                response=f"SQL Agent: An error occurred while processing your database query: {str(e)}",
                query_type=QueryType.DATABASE_ONLY.value,
                success=False,
                type="error",
                error=str(e)
            )
        )