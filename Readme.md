````markdown
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

## Flow Diagram
<!-- ![Agent Flow Diagram](link) -->



## Screenshots
<!-- chatinterface
![Chat Interface](https://raw.githubusercontent.com/0rajnishk/AgentFlow/main/screenshots/chat_interface.png) -->

<!-- admin
![Admin Interface](https://raw.githubusercontent.com/0rajnishk/AgentFlow/main/screenshots/admin_interface.png) -->

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
python agents/intent_classifier.py
python agents/error_handler.py
```

## Project Structure

```text
.
├── agents/
│   ├── sql_agent.py
│   ├── document_agent.py
│   ├── hybrid_agent.py
│   ├── intent_classifier.py
│   ├── error_handler.py
├── data/
│   └── project.db
├── utils/
│   ├── sql_utils.py
│   ├── embedding_utils.py
├── README.md
└── requirements.txt
```

## Example Query Flow

1. User asks: *"Show me the top 5 most expensive products."*
2. Intent Classifier routes to SQL Agent.
3. SQL Agent generates SQL using Gemini, executes on `project.db`, and summarizes.
4. Response returned to the user through the intent classifier.



