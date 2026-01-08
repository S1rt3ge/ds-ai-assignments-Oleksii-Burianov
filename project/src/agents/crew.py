import logging
from typing import Optional

from crewai import Agent, Crew, Task, Process

from src.agents.definitions import (
    create_planner_agent,
    create_retrieval_agent,
    create_synthesis_agent,
)
from src.agents.models import ResearchResult, RetrievalSource
from src.agents.tools import RAGSearchTool
from src.routing.router import QueryRouter
from src.rag.pipeline import RAGPipeline
from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)


class ResearchCrew:
    """Multi-agent crew for research tasks."""

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "mistral:latest",
        router: Optional[QueryRouter] = None,
        pipeline: Optional[RAGPipeline] = None,
        verbose: bool = True,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.router = router or QueryRouter()
        self.pipeline = pipeline or RAGPipeline()
        self.verbose = verbose
        self._llm = None
        self._agents = {}
        self._setup_agents()

    def _get_crewai_llm(self):
        """Get LLM configuration for crewAI."""
        if self.llm_provider == "ollama":
            return f"ollama/{self.llm_model}"
        elif self.llm_provider == "openrouter":
            return f"openrouter/{self.llm_model}"
        return self.llm_model

    def _setup_agents(self):
        """Initialize all agents."""
        llm = self._get_crewai_llm()
        self._agents["planner"] = create_planner_agent(llm=llm, router=self.router)
        self._agents["retrieval"] = create_retrieval_agent(llm=llm, pipeline=self.pipeline)
        self._agents["synthesis"] = create_synthesis_agent(llm=llm)

    def _create_planning_task(self, query: str) -> Task:
        """Create the planning task."""
        return Task(
            description=(
                f"Analyze the following query and determine the research strategy:\n\n"
                f"Query: {query}\n\n"
                f"Use the query_analysis tool to analyze complexity and type.\n"
                f"Determine if RAG retrieval is needed based on:\n"
                f"- Complexity score >= 30 suggests RAG may help\n"
                f"- Questions requiring analysis or reasoning benefit from RAG\n"
                f"- Simple factual questions may not need RAG\n\n"
                f"Provide your assessment including:\n"
                f"1. Whether RAG is needed (needs_rag: true/false)\n"
                f"2. The complexity score\n"
                f"3. Your research strategy"
            ),
            expected_output=(
                "A research plan containing:\n"
                "- needs_rag: boolean indicating if document retrieval is needed\n"
                "- complexity_score: integer 0-100\n"
                "- strategy: description of how to approach this query\n"
                "- question_type: the type of question detected"
            ),
            agent=self._agents["planner"],
        )

    def _create_retrieval_task(self, query: str, planning_task: Task) -> Task:
        """Create the retrieval task."""
        return Task(
            description=(
                f"Based on the planning analysis, search for relevant information.\n\n"
                f"Query: {query}\n\n"
                f"Use the rag_search tool to find relevant documents.\n"
                f"If the planner indicated needs_rag=false, you may skip retrieval.\n"
                f"Return the retrieved context with source information."
            ),
            expected_output=(
                "Retrieved context containing:\n"
                "- Relevant text passages from documents\n"
                "- Source citations (filename, chunk index)\n"
                "- Relevance scores\n"
                "If no retrieval needed, state that clearly."
            ),
            agent=self._agents["retrieval"],
            context=[planning_task],
        )

    def _create_synthesis_task(
        self, query: str, planning_task: Task, retrieval_task: Task
    ) -> Task:
        """Create the synthesis task."""
        return Task(
            description=(
                f"Create a comprehensive answer to the user's query.\n\n"
                f"Query: {query}\n\n"
                f"Use the planning analysis and retrieved context to formulate your answer.\n"
                f"If context was retrieved, include inline citations like [Source 1: filename].\n"
                f"If no context was available, provide a direct answer based on your knowledge.\n"
                f"Be accurate, concise, and well-structured."
            ),
            expected_output=(
                "A well-structured answer containing:\n"
                "- Direct response to the query\n"
                "- Inline citations if sources were used\n"
                "- Clear, accurate information"
            ),
            agent=self._agents["synthesis"],
            context=[planning_task, retrieval_task],
        )

    def _should_skip_retrieval(self, planning_output: str) -> bool:
        """Determine if retrieval should be skipped based on planning output."""
        planning_lower = planning_output.lower()
        if "needs_rag: false" in planning_lower or "needs_rag=false" in planning_lower:
            return True
        if "no rag" in planning_lower or "skip retrieval" in planning_lower:
            return True
        return False

    def _has_indexed_documents(self) -> bool:
        """Check if there are indexed documents available."""
        return self.pipeline.get_indexed_count() > 0

    def research(self, query: str) -> ResearchResult:
        """Execute the research workflow."""
        logger.info(f"Starting research for query: {query}")

        planning_task = self._create_planning_task(query)
        retrieval_task = self._create_retrieval_task(query, planning_task)
        synthesis_task = self._create_synthesis_task(query, planning_task, retrieval_task)

        tasks = [planning_task, retrieval_task, synthesis_task]

        if not self._has_indexed_documents():
            logger.info("No indexed documents, using simplified workflow")
            tasks = [planning_task, synthesis_task]
            synthesis_task.context = [planning_task]

        crew = Crew(
            agents=list(self._agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
            memory=False,
        )

        result = crew.kickoff()

        return self._parse_result(query, result, tasks)

    def _parse_result(self, query: str, crew_result: any, tasks: list) -> ResearchResult:
        """Parse crew result into ResearchResult model."""
        answer = str(crew_result)
        planning_output = ""
        if tasks and hasattr(tasks[0], "output") and tasks[0].output:
            planning_output = str(tasks[0].output)

        complexity_score = self._extract_complexity(planning_output)
        used_rag = len(tasks) > 2 and not self._should_skip_retrieval(planning_output)
        strategy = self._extract_strategy(planning_output)
        citations = self._extract_citations(answer)

        return ResearchResult(
            query=query,
            answer=answer,
            citations=citations,
            used_rag=used_rag,
            complexity_score=complexity_score,
            strategy=strategy,
            sources=[],
        )

    def _extract_complexity(self, planning_output: str) -> int:
        """Extract complexity score from planning output."""
        import re
        match = re.search(r"complexity[_\s]?score[:\s]+(\d+)", planning_output.lower())
        if match:
            return min(100, max(0, int(match.group(1))))
        return 50

    def _extract_strategy(self, planning_output: str) -> str:
        """Extract strategy from planning output."""
        if "strategy:" in planning_output.lower():
            parts = planning_output.lower().split("strategy:")
            if len(parts) > 1:
                return parts[1].split("\n")[0].strip()[:200]
        return "General research approach"

    def _extract_citations(self, answer: str) -> list:
        """Extract citations from the answer."""
        import re
        citations = re.findall(r"\[Source \d+:[^\]]+\]", answer)
        return list(set(citations))

    def index_document(self, file_path: str, chunking_strategy: str = "fixed") -> int:
        """Index a document for RAG retrieval."""
        return self.pipeline.index_document(file_path, chunking_strategy)

    def clear_index(self):
        """Clear all indexed documents."""
        self.pipeline.clear_index()

    def get_indexed_count(self) -> int:
        """Get number of indexed chunks."""
        return self.pipeline.get_indexed_count()
