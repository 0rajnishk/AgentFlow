# app/services/agent_orchestrator.py

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from app.services.intent_classifier import IntentClassifier, QueryType
from app.services.nl2sql import DatabaseManager, NLToSQLConverter, SQLAnswerGenerator
from app.llm.embeddings import load_vectorstore
from app.llm.prompts import get_gemini_response
from app.core.logging import logger
import google.generativeai as genai
from app.core.config import get_settings

# --- Access Control Configuration ---
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "Admin": ["all_access"],
    "User": ["document_access", "basic_database_access", "region_access:Global"],
    "Finance": ["document_access", "basic_database_access", "finance_access", "region_access:Global"],
    "Planning": ["document_access", "basic_database_access", "planning_access", "region_access:Global"],
}

class AgentOrchestrator:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.db_manager = DatabaseManager()
        self.nl2sql_converter = NLToSQLConverter()
        self.sql_answer_generator = SQLAnswerGenerator()
        
        # Initialize Gemini model for streaming
        genai.configure(api_key=get_settings().GENAI_API_KEY)
        self.model = genai.GenerativeModel(get_settings().GENAI_MODEL)
        
        # Load vector store once
        self.store = load_vectorstore()
        logger.info("AgentOrchestrator initialized successfully")
    
    def _check_access(self, user_role: str, user_region: str, required_permissions: List[str]) -> bool:
        """Check if the user's role and region grant the required permissions."""
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        
        if "all_access" in user_permissions:
            return True

        for req_perm in required_permissions:
            if req_perm.startswith("region_access:"):
                parts = req_perm.split(":")
                if len(parts) < 2:
                    logger.error(f"Malformed region_access permission: '{req_perm}'")
                    return False
                requested_region = parts[1]
                
                if (f"region_access:{requested_region}" not in user_permissions and
                    "region_access:Global" not in user_permissions):
                    logger.warning(f"Access denied for role '{user_role}' to region '{requested_region}'")
                    return False
            elif req_perm not in user_permissions:
                logger.warning(f"Access denied for role '{user_role}'. Missing permission: '{req_perm}'")
                return False
        return True

    async def process_query_stream(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Main orchestration method that streams responses as they're generated."""
        user_context = user_context if user_context is not None else {}
        user_role = user_context.get("role_name", "User")
        user_region = user_context.get("region", "Global")

        logger.info(f"Processing streaming query: '{query}' for user: {user_context.get('username', 'Anonymous')} (Role: {user_role})")

        try:
            # Step 1: Classify the query intent
            classification = self.intent_classifier.classify_query(query, user_context)
            query_type = classification['query_type']
            required_permissions = classification.get("required_permissions", [])
            
            # Send classification info immediately
            yield {
                "type": "classification",
                "query_type": query_type,
                "classification": classification,
                "required_permissions": required_permissions
            }
            
            # Step 2: Access Control Check
            if not self._check_access(user_role, user_region, required_permissions):
                yield {
                    "type": "error",
                    "error": "Access Denied: You do not have permission to view this data or perform this action.",
                    "query": query,
                    "query_type": query_type
                }
                return

            # Step 3: Route to appropriate streaming handler
            if query_type == QueryType.DOCUMENT_ONLY.value:
                async for chunk in self._handle_document_query_stream(query, classification, user_context):
                    yield chunk
            
            elif query_type == QueryType.DATABASE_ONLY.value:
                async for chunk in self._handle_database_query_stream(query, classification, user_context):
                    yield chunk
            
            elif query_type == QueryType.HYBRID.value:
                async for chunk in self._handle_hybrid_query_stream(query, classification, user_context):
                    yield chunk
            
            else:  # UNCLEAR
                async for chunk in self._handle_unclear_query_stream(query, classification, user_context):
                    yield chunk
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "query": query,
                "response": "I apologize, but I encountered an error while processing your query. Please try rephrasing your question or contact support if the issue persists."
            }

    async def _stream_gemini_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from Gemini API with enhanced error handling"""
        try:
            logger.info("Starting Gemini streaming response")
            
            # Use Gemini's streaming API
            response = self.model.generate_content(prompt, stream=True)
            
            chunk_count = 0
            for chunk in response:
                if chunk.text:
                    chunk_count += 1
                    logger.debug(f"Streaming chunk {chunk_count}: {len(chunk.text)} characters")
                    yield chunk.text
                    
            logger.info(f"Completed streaming {chunk_count} chunks")
                    
        except Exception as e:
            logger.error(f"Error streaming from Gemini: {e}")
            yield f"\n\n**Error generating response:** {str(e)}\n\nPlease try rephrasing your question or contact support if this issue persists."

    async def _handle_document_query_stream(self, query: str, classification: Dict[str, Any], user_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle document queries with enhanced streaming and prompts"""
        try:
            # Send status update
            yield {
                "type": "status",
                "message": "🔍 Searching through policy documents..."
            }
            
            if not self.store:
                yield {
                    "type": "error",
                    "error": "Document store not available",
                    "response": "I don't have access to policy documents at the moment. Please ensure documents are uploaded and indexed."
                }
                return
            
            # Search for relevant documents with enhanced parameters
            results = self.store.similarity_search(query, k=5)
            
            if not results:
                yield {
                    "type": "error",
                    "error": "No relevant documents found",
                    "response": f"I couldn't find any policy documents relevant to your question: '{query}'. Please try rephrasing your question or ask about a different topic."
                }
                return
            
            # Send document sources
            document_sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
            yield {
                "type": "document_sources",
                "sources": document_sources
            }
            
            # Send status update
            yield {
                "type": "status",
                "message": f"📄 Found {len(results)} relevant documents, generating comprehensive answer..."
            }
            
            # Create enhanced prompt for streaming
            context = "\n\n---\n\n".join([f"Document: {doc.metadata.get('title', 'Policy Document')}\nContent: {doc.page_content}" for doc in results])
            
            user_role = user_context.get("role_name", "User")
            
            prompt = f"""You are a knowledgeable supply chain policy expert at Syngenta. A {user_role} has asked a question about company policies and procedures.

**User Question:** "{query}"

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
            
            # Stream the response
            full_response = ""
            async for chunk in self._stream_gemini_response(prompt):
                full_response += chunk
                yield {
                    "type": "response_chunk",
                    "chunk": chunk,
                    "query_type": "document_only"
                }

            
            # Send final response
            yield {
                "type": "response_complete",
                "query": query,
                "response": full_response,
                "query_type": "document_only",
                "classification": classification,
                "document_sources": document_sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error handling document query: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "response": f"I encountered an error while searching through the documents: {str(e)}. Please try rephrasing your question."
            }

    async def _handle_database_query_stream(self, query: str, classification: Dict[str, Any], user_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle database queries with enhanced streaming and error handling"""
        try:
            # Send status update
            yield {
                "type": "status",
                "message": "🔄 Converting your question to SQL query..."
            }
            
            # Convert to SQL with user context - FIXED: Now passing user_context correctly
            sql_query = self.nl2sql_converter.convert_to_sql(query, user_context)
            
            yield {
                "type": "sql_generated",
                "sql_query": sql_query
            }
            
            # Send status update
            yield {
                "type": "status",
                "message": "📊 Executing database query..."
            }
            
            # Execute SQL with enhanced error handling
            try:
                query_results = self.db_manager.execute_query(sql_query)
            except Exception as db_error:
                logger.error(f"Database execution error: {db_error}")
                yield {
                    "type": "error",
                    "error": f"Database query failed: {str(db_error)}",
                    "response": f"I encountered an error while executing the database query. This might be due to:\n\n- Complex query requirements\n- Data access restrictions\n- Temporary database issues\n\nPlease try rephrasing your question or ask for a simpler analysis.\n\n**Generated Query:** `{sql_query}`"
                }
                return
            
            yield {
                "type": "data_results",
                "data_results": query_results,
                "result_count": len(query_results)
            }
            
            # Send status update
            yield {
                "type": "status",
                "message": f"✨ Analyzing {len(query_results)} records and generating insights..."
            }
            
            # Generate streaming response using the enhanced SQL answer generator
            full_response = self.sql_answer_generator.generate_answer(query, sql_query, query_results, user_context)
            
            # Stream the response character by character for real-time feel
            for i in range(0, len(full_response), 10):  # Send in chunks of 10 characters
                chunk = full_response[i:i+10]
                yield {
                    "type": "response_chunk",
                    "chunk": chunk,
                    "query_type": "database_only"
                }
                # Small delay for streaming effect
                await asyncio.sleep(0.01)
            
            # Send final response
            yield {
                "type": "response_complete",
                "query": query,
                "response": full_response,
                "query_type": "database_only",
                "classification": classification,
                "sql_query": sql_query,
                "data_results": query_results,
                "result_count": len(query_results),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error handling database query: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "response": f"I encountered an error while analyzing the database: {str(e)}. Please try rephrasing your question or ask for a simpler query."
            }
