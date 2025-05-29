from fastapi import APIRouter, File, UploadFile
from typing import List
from app.services.document import DocumentAPI
from app.core.logging import logger

router = APIRouter()
doc = DocumentAPI()

@router.get("/{document_id}")
def get_document(document_id: str):
    logger.info(f"Retrieving document with ID: {document_id}")
    document = doc.get_document(document_id)
    if document is None:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}
    logger.info(f"Document with ID '{document_id}' retrieved successfully.")
    return document

@router.post("/")
def upload_document(files: List[UploadFile] = File(...)):
    logger.info(f"Uploading {len(files)} file(s): {[file.filename for file in files]}")
    doc.upload_document(files)
    return {"message": "Upload document", "files": [file.filename for file in files]}

@router.get("/")
def get_all_documents():
    logger.info("Retrieving all documents.")
    documents = doc.get_all_documents()
    if not documents:
        logger.warning("No documents found.")
        return {"message": "No documents found"}
    logger.info(f"Retrieved {len(documents)} documents.")
    return documents

@router.delete("/{document_id}")
def delete_document(document_id: str):
    logger.info(f"Deleting document with ID: {document_id}")
    result = doc.delete_document(document_id)
    if result:
        logger.info(f"Document with ID '{document_id}' deleted successfully.")
        return {"message": f"Document {document_id} deleted"}
    else:
        logger.warning(f"Document with ID '{document_id}' not found.")
        return {"message": f"Document {document_id} not found"}