from typing import List, Optional
from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    """Output model for the Planner Agent."""
    needs_rag: bool = Field(description="Whether RAG retrieval is needed")
    complexity_score: int = Field(ge=0, le=100, description="Query complexity score")
    strategy: str = Field(description="Research strategy to follow")
    question_type: str = Field(description="Type of question detected")
    reasoning: str = Field(description="Explanation for the planning decision")


class RetrievalSource(BaseModel):
    """A single source from RAG retrieval."""
    filename: str
    chunk_index: int
    text: str
    score: float


class RetrievalOutput(BaseModel):
    """Output model for the Retrieval Agent."""
    sources: List[RetrievalSource] = Field(default_factory=list)
    context: str = Field(description="Formatted context from retrieved sources")
    source_count: int = Field(ge=0, description="Number of sources retrieved")


class SynthesisOutput(BaseModel):
    """Output model for the Synthesis Agent."""
    answer: str = Field(description="Final answer with inline citations")
    citations: List[str] = Field(default_factory=list, description="List of cited sources")
    confidence: str = Field(description="Confidence level: high, medium, low")


class ResearchState(BaseModel):
    """Shared state between agents in the workflow."""
    query: str
    complexity_score: int = 0
    question_type: str = ""
    needs_rag: bool = False
    strategy: str = ""
    retrieved_context: str = ""
    citations: List[str] = Field(default_factory=list)
    final_answer: str = ""
    human_approved: bool = True


class ResearchResult(BaseModel):
    """Final result from the research assistant."""
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    used_rag: bool
    complexity_score: int
    strategy: str
    sources: List[RetrievalSource] = Field(default_factory=list)
