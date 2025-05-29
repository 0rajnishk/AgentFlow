import google.generativeai as genai
from app.core.config import get_settings
from app.core.logging import logger

genai.configure(api_key=get_settings().GENAI_API_KEY)

def get_gemini_response(query, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    model = genai.GenerativeModel(get_settings().GENAI_MODEL)
    prompt = f"""
        You are a helpful assistant designed to support a user working in the Syngenta company and they might me ask query related to uploaded policy doucments or business related queries about sales, revenue etc.
        Use the following context documents to answer the user’s question as clearly and accurately as possible.

        ---

        ### Context:
        {context}

        ---

        ### Query:
        {query}

        ---

        ### Rules for Response:

        1. Provide a concise and informative answer based on the context.
        2. If the context does not contain sufficient information to answer the question, respond with warmly and politely that you cannot answer the question.
        3. Avoid using any external knowledge or information not provided in the context.
        4. Do not include any disclaimers or unnecessary information.
        5. Do not repeat the question or context in your answer.
        6. Use a friendly and professional tone.
        7. If the context contains multiple relevant pieces of information, summarize them in a coherent manner.
        8. If the context is too long, focus on the most relevant parts to answer the question.
        9. If the context is in a different language, respond in the same language as the context.
        10. If the context contains conflicting information, provide a balanced view and mention the discrepancies.
        11. If the context contains technical terms or jargon, explain them in simple terms.
        12. If the context contains a list of items, provide a summary of the list in your answer.
        13. If the context contains a question, answer it directly without rephrasing.
        14. If the context contains a quote, provide the quote in quotation marks.
        15. If the context contains a date or time, provide it in a clear and understandable format.
        16. If the context contains a location, provide it in a clear and understandable format.
        17. If the context contains a name, provide it in a clear and understandable format.
        18. If the context contains a number, provide it in a clear and understandable format.

        ### Answer format:
        Try to give anser in markdown format. 

        ---

        ### Answer:
        """

    try:
        logger.info("Sending prompt to Gemini model.")
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini model.")
        return response.text
    except Exception as e:
        logger.error(f"Error processing Gemini request: {e}")
        return "Error processing Gemini request"