# (Contents of fastapi_app.py remain unchanged from previous response)
import asyncio
import os
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up basic logging for FastAPI
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- FastAPI Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

app = FastAPI(title="Agent-Powered Query API")

# File paths for inter-process communication
QUERY_FILE_PATH = "query_to_agents.txt"
RESPONSE_FILE_PATH = "response_from_agents.txt"

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    request_id = str(uuid.uuid4()) # Generate a unique ID for this request
    logging.info(f"Received API query: '{request.query}' (Request ID: {request_id})")

    # 1. Write query to file for agents to pick up
    try:
        # Use 'w' mode to overwrite, ensuring only one active query at a time for simplicity
        with open(QUERY_FILE_PATH, "w") as f:
            f.write(f"{request_id}:::{request.query}")
        logging.info(f"Wrote query to {QUERY_FILE_PATH} for Request ID: {request_id}")
    except Exception as e:
        logging.error(f"Error writing query to file: {e}")
        raise HTTPException(status_code=500, detail="Failed to send query to agents.")

    # 2. Wait for response from agents via file
    response_received = False
    max_retries = 60 # Check for up to 60 seconds (60 * 1 second sleep)
    for _ in range(max_retries):
        if os.path.exists(RESPONSE_FILE_PATH):
            with open(RESPONSE_FILE_PATH, "r+") as f:
                response_line = f.readline().strip()
                if response_line:
                    try:
                        # Expecting "request_id:::response_text"
                        resp_req_id, response_text = response_line.split(":::", 1)
                        if resp_req_id == request_id: # Check if it's the response for *this* request
                            f.truncate(0) # Clear the file after reading
                            logging.info(f"Read response from {RESPONSE_FILE_PATH} for Request ID: {request_id}")
                            response_received = True
                            return QueryResponse(response=response_text)
                        else:
                            # It's a response for a different request, or a stale one.
                            # Leave it for now, let the next request pick it up if it's theirs,
                            # or it will be overwritten by a new query.
                            logging.warning(f"Found response for ID '{resp_req_id}', but expected '{request_id}'. Leaving file.")
                    except ValueError:
                        logging.error(f"Invalid response format in file: {response_line}")
                else:
                    # File is empty, wait for content
                    pass
        await asyncio.sleep(1) # Wait 1 second before checking again

    # If loop finishes without returning, it's a timeout
    logging.error(f"Timeout waiting for agent response for Request ID: {request_id}")
    # Clear the query file in case it was stuck
    if os.path.exists(QUERY_FILE_PATH):
        with open(QUERY_FILE_PATH, "w") as f:
            f.truncate(0)
    raise HTTPException(status_code=504, detail="Agent response timed out.")


if __name__ == "__main__":
    import uvicorn
    # Clean up old files if they exist on startup
    if os.path.exists(QUERY_FILE_PATH):
        os.remove(QUERY_FILE_PATH)
    if os.path.exists(RESPONSE_FILE_PATH):
        os.remove(RESPONSE_FILE_PATH)

    logging.info("Starting FastAPI application (separate process)...")
    uvicorn.run(app, host="0.0.0.0", port=9000)