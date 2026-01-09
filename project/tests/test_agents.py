import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from src.agents import create_research_assistant

assistant = create_research_assistant(
    llm_provider="ollama",
    llm_model="ministral-3:3b",
    verbose=False
)

result = assistant.query("What is deep learning?")
print(f"Answer: {result.answer}")
print(f"Used RAG: {result.used_rag}")
print(f"Complexity: {result.complexity_score}")