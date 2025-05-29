import asyncio
import os
import uuid
import logging
import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ─── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── FastAPI Setup ─────────────────────────────────────────────────
app = FastAPI(title="Agent Flow Query API", version="1.0.0")
router = APIRouter(prefix="/document", tags=["documents"])


# Mount static files (if you have JS, CSS, etc. later)
app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# ─── File Paths ────────────────────────────────────────────────────
QUERY_FILE_PATH = "query_to_agents.txt"
RESPONSE_FILE_PATH = "response_from_agents.txt"
DB_FOLDER = "vectorstores"
os.makedirs(DB_FOLDER, exist_ok=True)

# ─── Google Gemini Embeddings ──────────────────────────────────────
GENAI_API_KEY = "AIzaSyAyau1UaTUWYDdYTKz37zzU94zhFhddzuA"  # Export this in your environment
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GENAI_API_KEY
)

# ─── PDF Processor ─────────────────────────────────────────────────
def process_pdf(pdf_path: str, pdf_id: str):
    logger.info(f"Starting PDF processing for: {pdf_path} with id: {pdf_id}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from PDF.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(os.path.join(DB_FOLDER, pdf_id))
    logger.info(f"Vectorstore saved locally with id: {pdf_id}")
    return {"success": True, "message": "PDF processed successfully"}

# ─── In-Memory Document Manager ────────────────────────────────────
class DocManager:
    def __init__(self):
        self.docs = {}  # doc_id: {filename, path}

    def upload_document(self, files: List[UploadFile]):
        for f in files:
            doc_id = str(uuid.uuid4())
            save_path = os.path.join(DB_FOLDER, f"{doc_id}_{f.filename}")
            with open(save_path, "wb") as dest:
                shutil.copyfileobj(f.file, dest)
            process_pdf(save_path, doc_id)
            self.docs[doc_id] = {"filename": f.filename, "path": save_path}

    def get_document(self, doc_id: str):
        return self.docs.get(doc_id)

    def get_all_documents(self):
        return self.docs

    def delete_document(self, doc_id: str):
        meta = self.docs.pop(doc_id, None)
        if meta:
            try:
                os.remove(meta["path"])
                shutil.rmtree(os.path.join(DB_FOLDER, doc_id), ignore_errors=True)
            except Exception:
                pass
        return meta is not None

doc = DocManager()



# ─── Index Route ───────────────────────────────────────────────────
@app.get("/")
def get_index():
    return FileResponse("static/index.html")

# # ─── Admin Route ───────────────────────────────────────────────────
# @app.get("/admin")
# def get_admin():
#     return FileResponse("static/admin.html")


# ─── Document Endpoints ────────────────────────────────────────────
@router.post("/")
async def upload_document(files: List[UploadFile] = File(...)):
    logger.info(f"Uploading {len(files)} file(s): {[file.filename for file in files]}")
    doc.upload_document(files)
    return {"message": "Upload document", "files": [file.filename for file in files]}

@router.get("/{document_id}")
async def get_document(document_id: str):
    logger.info(f"Retrieving document with ID: {document_id}")
    document = doc.get_document(document_id)
    if document is None:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}
    logger.info(f"Document with ID '{document_id}' retrieved successfully.")
    return document

@router.get("/")
async def get_all_documents():
    logger.info("Retrieving all documents.")
    documents = doc.get_all_documents()
    if not documents:
        logger.warning("No documents found.")
        return {"message": "No documents found"}
    logger.info(f"Retrieved {len(documents)} documents.")
    return documents

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    logger.info(f"Deleting document with ID: {document_id}")
    result = doc.delete_document(document_id)
    if result:
        logger.info(f"Document with ID '{document_id}' deleted successfully.")
        return {"message": f"Document {document_id} deleted"}
    else:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}


# ─── Query Processing Endpoint ─────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    logging.info(f"Received API query: '{request.query}' (Request ID: {request_id})")

    try:
        with open(QUERY_FILE_PATH, "w") as f:
            f.write(f"{request_id}:::{request.query}")
        logging.info(f"Wrote query to {QUERY_FILE_PATH} for Request ID: {request_id}")
    except Exception as e:
        logging.error(f"Error writing query to file: {e}")
        raise HTTPException(status_code=500, detail="Failed to send query to agents.")

    response_received = False
    max_retries = 60

    for _ in range(max_retries):
        if os.path.exists(RESPONSE_FILE_PATH):
            with open(RESPONSE_FILE_PATH, "r+", encoding="utf-8") as f:
                content = f.read()
                try:
                    # split on the first ::: only, allowing the rest to be multiline text
                    resp_req_id, response_text = content.split(":::", 1)
                    if resp_req_id.strip() == request_id:
                        f.seek(0)
                        f.truncate()
                        logging.info(f"Read full response from {RESPONSE_FILE_PATH} for Request ID: {request_id}")
                        response_received = True
                        return QueryResponse(response=response_text.strip())
                    else:
                        logging.warning(f"Found response for ID '{resp_req_id.strip()}', but expected '{request_id}'. Leaving file.")
                except ValueError:
                    logging.error("Invalid response format in file. Expected 'request_id:::response_text'")
        await asyncio.sleep(1)

    logging.error(f"Timeout waiting for agent response for Request ID: {request_id}")
    if os.path.exists(QUERY_FILE_PATH):
        with open(QUERY_FILE_PATH, "w") as f:
            f.truncate(0)
    raise HTTPException(status_code=504, detail="Agent response timed out.")





# ─── Mount Document Router ─────────────────────────────────────────
app.include_router(router)

# ─── Run Server ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    if os.path.exists(QUERY_FILE_PATH):
        os.remove(QUERY_FILE_PATH)
    if os.path.exists(RESPONSE_FILE_PATH):
        os.remove(RESPONSE_FILE_PATH)

    logging.info("Starting FastAPI application (separate process)...")
    uvicorn.run(app, host="0.0.0.0", port=9000)







# # (Contents of fastapi_app.py remain unchanged from previous response)
# import asyncio
# import os
# import uuid
# import logging
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # Set up basic logging for FastAPI
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # --- FastAPI Models ---
# class QueryRequest(BaseModel):
#     query: str

# class QueryResponse(BaseModel):
#     response: str

# app = FastAPI(title="Agent-Powered Query API")

# # File paths for inter-process communication
# QUERY_FILE_PATH = "query_to_agents.txt"
# RESPONSE_FILE_PATH = "response_from_agents.txt"

# @app.post("/query", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     request_id = str(uuid.uuid4()) # Generate a unique ID for this request
#     logging.info(f"Received API query: '{request.query}' (Request ID: {request_id})")

#     # 1. Write query to file for agents to pick up
#     try:
#         # Use 'w' mode to overwrite, ensuring only one active query at a time for simplicity
#         with open(QUERY_FILE_PATH, "w") as f:
#             f.write(f"{request_id}:::{request.query}")
#         logging.info(f"Wrote query to {QUERY_FILE_PATH} for Request ID: {request_id}")
#     except Exception as e:
#         logging.error(f"Error writing query to file: {e}")
#         raise HTTPException(status_code=500, detail="Failed to send query to agents.")

#     # 2. Wait for response from agents via file
#     response_received = False
#     max_retries = 60 # Check for up to 60 seconds (60 * 1 second sleep)
#     for _ in range(max_retries):
#         if os.path.exists(RESPONSE_FILE_PATH):
#             with open(RESPONSE_FILE_PATH, "r+") as f:
#                 response_line = f.readline().strip()
#                 if response_line:
#                     try:
#                         # Expecting "request_id:::response_text"
#                         resp_req_id, response_text = response_line.split(":::", 1)
#                         if resp_req_id == request_id: # Check if it's the response for *this* request
#                             f.truncate(0) # Clear the file after reading
#                             logging.info(f"Read response from {RESPONSE_FILE_PATH} for Request ID: {request_id}")
#                             response_received = True
#                             return QueryResponse(response=response_text)
#                         else:
#                             # It's a response for a different request, or a stale one.
#                             # Leave it for now, let the next request pick it up if it's theirs,
#                             # or it will be overwritten by a new query.
#                             logging.warning(f"Found response for ID '{resp_req_id}', but expected '{request_id}'. Leaving file.")
#                     except ValueError:
#                         logging.error(f"Invalid response format in file: {response_line}")
#                 else:
#                     # File is empty, wait for content
#                     pass
#         await asyncio.sleep(1) # Wait 1 second before checking again

#     # If loop finishes without returning, it's a timeout
#     logging.error(f"Timeout waiting for agent response for Request ID: {request_id}")
#     # Clear the query file in case it was stuck
#     if os.path.exists(QUERY_FILE_PATH):
#         with open(QUERY_FILE_PATH, "w") as f:
#             f.truncate(0)
#     raise HTTPException(status_code=504, detail="Agent response timed out.")


# if __name__ == "__main__":
#     import uvicorn
#     # Clean up old files if they exist on startup
#     if os.path.exists(QUERY_FILE_PATH):
#         os.remove(QUERY_FILE_PATH)
#     if os.path.exists(RESPONSE_FILE_PATH):
#         os.remove(RESPONSE_FILE_PATH)

#     logging.info("Starting FastAPI application (separate process)...")
#     uvicorn.run(app, host="0.0.0.0", port=9000)