## Getting Started

1. Click the "Use this template" button at the top of this GitHub repository
2. Name the new repository `ds-ai-assignments-[user-name]` (e.g., `ds-ai-assignments-john-doe`)
3. Make the new repository private
4. Share access with assigned mentor(s)
5. Clone the new repository to a local machine

## How to Use

### Prerequisites

- Python 3.12+
- Poetry for dependency management

### Set Up the Virtual Environment

To create a virtual environment using Python 3.12, run:

```
poetry env use 3.12
```

### Install Dependencies

Install all project dependencies, including development tools:

```bash
poetry install
```

## Code Quality Tools

These tools are configured in `pyproject.toml` with a line length of 120 characters.


### Architecture

User Query -> Planner Agent -> Conditional Routing -> Retrieval Agent (if needed) -> Synthesis Agent -> Result

Workflow:
1. Planner analyzes query complexity and decides if RAG is needed
2. If needs_rag=true and documents are indexed, Retrieval agent searches documents
3. If needs_rag=false or no documents, skip directly to Synthesis
4. Synthesis agent generates final answer with citations

### Agents

Planner Agent:
- Role: Research Strategy Planner
- Tool: QueryAnalysisTool
- Task: Analyze query complexity, determine question type, decide if RAG needed

Retrieval Agent:
- Role: Information Retrieval Specialist
- Tool: RAGSearchTool
- Task: Search indexed documents, return relevant chunks with sources

Synthesis Agent:
- Role: Research Report Writer
- Tools: SummarizerTool, FactCheckerTool
- Task: Generate answer with citations, summarize text, verify facts

### Tools

QueryAnalysisTool - analyzes query complexity (0-100), detects question type
RAGSearchTool - semantic search over indexed documents
SummarizerTool - extracts key sentences from text
FactCheckerTool - verifies claims against documents

### Setup

1. Install:
```
poetry install
```

2. Create .env file:
```
OLLAMA_HOST=http://localhost:11434
OPENROUTER_API_KEY=your_key
```

3. Start Ollama:
```
ollama pull your_models
```

### Usage

Run UI:
```
cd project
poetry run streamlit run src/ui/app.py
```

Multi-Agent Mode in UI:
1. Enable "Multi-Agent System" toggle in sidebar
2. Select provider and model
3. Upload and index documents if needed
4. Enter query

### Demo Queries

Simple (no RAG):
```
What is machine learning?
```

Complex (uses RAG):
```
Analyze the methodology and findings in the uploaded research paper.
```

### Export

After running query, download results as MD or TXT from Export section.
