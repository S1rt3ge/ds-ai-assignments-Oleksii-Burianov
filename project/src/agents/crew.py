import logging
from typing import Optional

from crewai import Agent, Crew, Task, Process

from src.agents.definitions import (
    create_planner_agent,
    create_retrieval_agent,
    create_synthesis_agent,
)
from src.agents.models import ResearchResult, ResearchState
from src.routing.router import QueryRouter
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ResearchCrew:
    """Multi-agent crew for research tasks."""

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "ministral-3:3b",
        router: Optional[QueryRouter] = None,
        pipeline: Optional[RAGPipeline] = None,
        verbose: bool = False,
        human_in_the_loop: bool = False,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.router = router or QueryRouter()
        self.pipeline = pipeline or RAGPipeline()
        self.verbose = verbose
        self.human_in_the_loop = human_in_the_loop
        self._llm = None
        self._agents = {}
        self._state: Optional[ResearchState] = None
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

    def _create_planning_task(self, query: str, human_input: bool = False) -> Task:
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
            human_input=human_input,
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

    def _run_planning_phase(self, query: str) -> str:
        """Run planning phase separately to get routing decision."""
        planning_task = self._create_planning_task(query, human_input=self.human_in_the_loop)
        planning_crew = Crew(
            agents=[self._agents["planner"]],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=False,
        )
        result = planning_crew.kickoff()
        return str(result)

    def _update_state_from_planning(self, planning_output: str):
        """Update shared state from planning output."""
        self._state.complexity_score = self._extract_complexity(planning_output)
        self._state.strategy = self._extract_strategy(planning_output)
        self._state.needs_rag = not self._should_skip_retrieval(planning_output)
        if "question_type:" in planning_output.lower():
            parts = planning_output.lower().split("question_type:")
            if len(parts) > 1:
                self._state.question_type = parts[1].split("\n")[0].strip()

    def research(self, query: str) -> ResearchResult:
        """Execute the research workflow with conditional routing."""
        logger.info(f"Starting research for query: {query}")
        logger.info(f"Indexed documents: {self._has_indexed_documents()}")
        logger.info(f"Human-in-the-loop: {self.human_in_the_loop}")

        self._state = ResearchState(query=query)

        planning_output = self._run_planning_phase(query)
        self._update_state_from_planning(planning_output)
        logger.info(f"Planning complete - needs_rag: {self._state.needs_rag}, complexity: {self._state.complexity_score}")

        has_docs = self._has_indexed_documents()
        should_retrieve = self._state.needs_rag and has_docs

        if should_retrieve:
            logger.info("Conditional routing: executing retrieval phase")
            tasks, agents = self._build_full_workflow(query, planning_output)
        else:
            reason = "no documents indexed" if not has_docs else "planner decided RAG not needed"
            logger.info(f"Conditional routing: skipping retrieval ({reason})")
            tasks, agents = self._build_synthesis_only_workflow(query, planning_output)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
            memory=False,
        )

        logger.info(f"Starting crew with {len(tasks)} tasks")
        result = crew.kickoff()

        self._state.final_answer = str(result)
        self._state.citations = self._extract_citations(str(result))

        return self._build_result(should_retrieve)

    def _build_full_workflow(self, query: str, planning_output: str):
        """Build workflow with retrieval."""
        retrieval_task = Task(
            description=(
                f"Search for relevant information for: {query}\n\n"
                f"Planning context: {planning_output}\n\n"
                f"Use the rag_search tool to find relevant documents."
            ),
            expected_output="Retrieved context with source citations.",
            agent=self._agents["retrieval"],
        )
        synthesis_task = Task(
            description=(
                f"Create a comprehensive answer for: {query}\n\n"
                f"Use the retrieved context to formulate your answer with citations."
            ),
            expected_output="Well-structured answer with inline citations.",
            agent=self._agents["synthesis"],
            context=[retrieval_task],
        )
        return [retrieval_task, synthesis_task], [self._agents["retrieval"], self._agents["synthesis"]]

    def _build_synthesis_only_workflow(self, query: str, planning_output: str):
        """Build workflow without retrieval."""
        synthesis_task = Task(
            description=(
                f"Create a comprehensive answer for: {query}\n\n"
                f"Planning context: {planning_output}\n\n"
                f"Provide a direct answer based on your knowledge."
            ),
            expected_output="Well-structured answer.",
            agent=self._agents["synthesis"],
        )
        return [synthesis_task], [self._agents["synthesis"]]

    def _build_result(self, used_rag: bool) -> ResearchResult:
        """Build final result from state."""
        return ResearchResult(
            query=self._state.query,
            answer=self._state.final_answer,
            citations=self._state.citations,
            used_rag=used_rag,
            complexity_score=self._state.complexity_score,
            strategy=self._state.strategy,
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
