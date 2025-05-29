from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import get_settings
from app.core.logging import logger
import os

UPLOAD_FOLDER = "uploads"
DB_FOLDER = "db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_settings().GENAI_API_KEY)

def process_pdf(pdf_path, pdf_id):
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

def load_vectorstore():
    stores = []
    for dir_name in os.listdir(DB_FOLDER):
        path = os.path.join(DB_FOLDER, dir_name)
        if os.path.isdir(path):
            try:
                logger.info(f"Attempting to load vectorstore from: {path}")
                store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
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