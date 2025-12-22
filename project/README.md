# Research Assistant

Multi-Agent Research System with Intelligent Routing & Prompting

## Installation


pip install poetry


poetry install


Install Ollama (for local models):

ollama pull gemma3:1b
ollama pull deepseek-r1:1.5b
ollama pull ministral-3:3b


## Run Application

poetry run streamlit run src/ui/app.py

Application opens at http://localhost:8501

### Supported Models

- **gemma3:1b** - Fast, basic quality (1B parameters)
- **deepseek-r1:1.5b** - Medium speed/quality (1.5B parameters)
- **ministral-3:3b** - Slow, high quality (3B parameters)
- 
### Prompting Strategies

1. **System Role** - Clear role definition with direct instructions
2. **Few-Shot** - Learning from examples before answering
3. **Chain-of-Thought** - Step-by-step reasoning
4. **Structured Output** - JSON-formatted responses

### Advanced Options

- **Temperature** (0.0-2.0)
  - Lower values (0.0) = More focused and deterministic responses
  - Higher values (2.0) = More creative and varied responses
  - Default: 0.7

- **Max Tokens** (100-4000)
  - Limits the maximum length of the generated response
  - Default: 1000

- **Streaming**
  - Enable real-time token generation (word-by-word display)
  - Default: Off

## Project Structure

```
project/
├── src/
│   ├── llm/              # LLM client implementation
│   ├── prompts/          # Prompt management
│   ├── ui/               # Streamlit interface
│   ├── routing/          # (Week 6)
│   ├── rag/              # (Week 7)
│   ├── agents/           # (Week 8)
│   └── tools/            # (Week 8)
├── data/                 # Dataset directory
├── tests/                # Unit tests
├── scripts/              # Utility scripts
└── pyproject.toml        # Poetry configuration
```
