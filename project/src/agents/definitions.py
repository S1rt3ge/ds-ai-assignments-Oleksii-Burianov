from typing import Optional

from crewai import Agent

from src.tools import QueryAnalysisTool, RAGSearchTool
from src.routing.router import QueryRouter
from src.rag.pipeline import RAGPipeline


def create_planner_agent(
    llm: any,
    router: Optional[QueryRouter] = None,
) -> Agent:
    """Create the Research Strategy Planner agent."""
    query_tool = QueryAnalysisTool(router=router)
    return Agent(
        role="Research Strategy Planner",
        goal="Analyze the user query and determine the optimal research approach",
        backstory=(
            "You are an expert in breaking down complex research questions. "
            "You analyze queries to determine their complexity, type, and what resources are needed. "
            "You create clear research strategies that guide the retrieval and synthesis process."
        ),
        tools=[query_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_retrieval_agent(
    llm: any,
    pipeline: Optional[RAGPipeline] = None,
) -> Agent:
    """Create the Information Retrieval Specialist agent."""
    rag_tool = RAGSearchTool(pipeline=pipeline)
    return Agent(
        role="Information Retrieval Specialist",
        goal="Find the most relevant information from indexed documents",
        backstory=(
            "You are an expert in semantic search and document retrieval. "
            "You excel at finding relevant passages from large document collections. "
            "You understand how to formulate effective search queries and select the best sources."
        ),
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_synthesis_agent(llm: any) -> Agent:
    """Create the Research Report Writer agent."""
    return Agent(
        role="Research Report Writer",
        goal="Create comprehensive answers with proper citations from the retrieved information",
        backstory=(
            "You are an expert in synthesizing information and academic writing. "
            "You create clear, well-structured answers that properly cite sources. "
            "You ensure accuracy by grounding responses in the provided context."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
