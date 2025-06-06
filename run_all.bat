@echo off
SETLOCAL

REM === Activate the virtual environment ===
CALL .\venv\Scripts\activate.bat

REM === Start agents in new terminals ===
start "SQL Agent" cmd /k "python agents\sql_agent.py"
start "Hybrid Agent" cmd /k "python agents\hybrid_agent.py"
start "Document Agent" cmd /k "python agents\document_agent.py"
start "Error Handler Agent" cmd /k "python agents\error_handler_agent.py"
start "Integration Agent" cmd /k "python agents\intent_classifier_agent.py"



REM === Start FastAPI server ===
start "FastAPI Server" cmd /k "uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

ENDLOCAL
