# Agent Flow

Agent Flow is a agent-based system built using the `uagents` framework. It combines document retrieval, SQL querying, and large language model reasoning to answer natural language questions. This system routes queries intelligently across different specialized agents to provide accurate, explainable, and context-aware responses.

## Overview

The system consists of several agents, each with a specific responsibility:

- Intent Classifier Agent: Determines the type of user query (SQL, document, hybrid, or unknown).
- SQL Agent: Translates natural language into SQL, executes it on a SQLite database (`project.db`), and summarizes the result.
- Document Agent: Performs similarity search across document embeddings to answer context-based questions.
- Hybrid Agent: Uses both similarity search and SQL execution for complex questions requiring multi-source reasoning.
- Error Handler Agent: Handles misclassified or ambiguous queries using LLM's general knowledge capabilities.

## Features

- Modular agent architecture using `uagents`.
- SQLite database (`project.db`) for structured supply chain data.
- LLM-powered SQL generation using Gemini.
- Document-based similarity search for unstructured data.
- Hybrid reasoning that combines structured and unstructured data.
- Fallback logic for gracefully handling unknown or poorly classified queries.

## Technologies Used

- Python
- uagents (Agent-based microservices framework)
- Google Gemini API (LLM for SQL and natural language reasoning)
- SQLite (Lightweight local database)
- Faiss (Vector store for similarity search)

## Architecture Diagrams

### Agent Workflow Chart
![Agents Workflow Chart](screenshots/agents-workflow-chart.png)

### FastAPI Workflow Chart
![FastAPI Workflow Chart](screenshots/fast-api-workflow-chart.png)

## Screenshots

### Chat Interface
![Chat Interface](screenshots/chat-interface.png)

### Admin Panel

#### Admin Login
![Admin Login](screenshots/admin-login.png)

#### Document Management
![Admin Documents](screenshots/admin-documents.png)

#### Document Upload
![Admin Document Upload](screenshots/admin-document-upload.png)

#### Database Upload
![Admin Database Upload](screenshots/admin-database-upload.png)

#### Database Connection Details
![Admin Database Connection Details](screenshots/admin-database-connection-details.png)

### AgentVerse Integration

#### AgentVerse Agent Details
![AgentVerse Agent Details](screenshots/agentverse-agent-details.png)

#### AgentVerse Local Agents
![AgentVerse Local Agents](screenshots/agentverse-local-agents.png)

#### AgentVerse Intent Agent Chat
![AgentVerse Intent Agent Chat](screenshots/agent-verse-intent-agent-chat.png)

#### AgentVerse Logs
![AgentVerse Logs](screenshots/agent-verse-logs.png)

### API Documentation

#### FastAPI Docs
![FastAPI Docs](screenshots/fast-api-docs.png)

#### Info Page
![Info Page](screenshots/info-page.png)

### System Logs
![Terminal Logs](screenshots/terminal-logs.png)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/0rajnishk/AgentFlow.git
cd agent-flow
```

### 2. Set Environment Variables

Create a `.env` file or export the following environment variable:

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the System

Launch each agent independently or as a managed process depending on your architecture. Example:

```bash
python agents/sql_agent.py
python agents/hybrid_agent.py
python agents/document_agent.py
python agents/error_handler_agent.py
```

### update the address of each of the above agents in the intent_classifier.py file
```python
# Replace these with the actual addresses printed by each worker agent at startup
SQL_AGENT_ADDR = "agent1qw4z53dh3ttmdqku5q0xpqm6s25a0g69fsgh3lr440ak38stvfcy79fh2lk"
HYBRID_AGENT_ADDR = "agent1q0sh6f3n2r8azrs524chrn0e7h7p3qkm25v502jzczkrgjmtnhe972h2g64"
DOC_AGENT_ADDR = "agent1q2tpmsy506wtsdn0j7823s2vdm7f50l485azzc6z8lh2zk50cqwevn03e6q"
ERROR_AGENT_ADDR = "agent1q08hrn7j6t7ywmwdllrvl903t08sn4xd2ua2t4hy4kd6uxspqq0rgaudrpx"

```


```bash
python agents/intent_classifier_agent.py
```


# Run them main.py our FastAPI server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```


## Example Query Flow

1. User asks: *"Show me the top 5 most expensive products."*
2. Intent Classifier routes to SQL Agent.
3. SQL Agent generates SQL using Gemini, executes on `project.db`, and summarizes.
4. Response returned to the user through the intent classifier.