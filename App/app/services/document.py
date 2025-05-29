import datetime
import os
import uuid
import json
from fastapi import UploadFile
from typing import Any, Dict, List, Optional
from app.core.logging import logger
from app.llm.embeddings import process_pdf

UPLOAD_DIR = "./UPLOAD_FOLDER"
MAPPING_FILE = os.path.join(UPLOAD_DIR, "file_mapping.json")

def load_mapping():
    if not os.path.exists(MAPPING_FILE):
        return {}
    with open(MAPPING_FILE, "r") as f:
        return json.load(f)

def save_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

class DocumentAPI:
    def __init__(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.mapping = load_mapping()

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        if document_id not in self.mapping:
            logger.warning(f"Document with id '{document_id}' not found.")
            return None
        filename = self.mapping[document_id]
        file_path = os.path.join(UPLOAD_DIR, document_id)
        if not os.path.exists(file_path):
            logger.warning(f"File '{file_path}' not found.")
            return None
        try:
            file_info = {
                "id": document_id,
                "original_filename": filename,
                "filepath": file_path,
                "size": os.path.getsize(file_path),
                "upload_time": datetime.datetime.fromtimestamp(os.path.getctime(file_path)),
                "status": "available"
            }
            return file_info
        except Exception as e:
            logger.error(f"Failed to retrieve document '{document_id}': {e}")
            return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        documents_list = []
        for doc_id, filename in self.mapping.items():
            file_path = os.path.join(UPLOAD_DIR, doc_id)
            if os.path.isfile(file_path):
                file_info = {
                    "id": doc_id,
                    "original_filename": filename,
                    "filepath": file_path,
                    "size": os.path.getsize(file_path),
                    "upload_time": datetime.datetime.fromtimestamp(os.path.getctime(file_path)),
                    "status": "available"
                }
                documents_list.append(file_info)
        return documents_list

    def upload_document(self, files: List[UploadFile]):
        if not files:
            logger.error("No files provided for upload.")
            return

        for file in files:
            ext = os.path.splitext(file.filename)[1]
            unique_id = str(uuid.uuid4()) + ext
            file_path = os.path.join(UPLOAD_DIR, unique_id)
            try:
                contents = file.file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)
                self.mapping[unique_id] = file.filename
                save_mapping(self.mapping)
                logger.info(f"Saved file: {file.filename} as {unique_id}")
                process_pdf(file_path, file.filename)
            except Exception as e:
                logger.error(f"Failed to upload file {file.filename}: {e}")
            finally:
                file.file.close()

    def delete_document(self, document_id: str) -> bool:
        file_path = os.path.join(UPLOAD_DIR, document_id)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                if document_id in self.mapping:
                    del self.mapping[document_id]
                    save_mapping(self.mapping)
                logger.info(f"File '{file_path}' deleted.")
                return True
            else:
                logger.warning(f"File '{file_path}' not found for deletion.")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file '{file_path}': {e}")
            return False
