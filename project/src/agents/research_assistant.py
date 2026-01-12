import logging
from typing import Optional, List

from src.agents.crew import ResearchCrew
from src.agents.models import ResearchResult
from src.routing.router import QueryRouter
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """Main interface for the multi-agent research assistant."""

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "ministral-3:3b",
        verbose: bool = False,
        human_in_the_loop: bool = False,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.verbose = verbose
        self.human_in_the_loop = human_in_the_loop
        self.router = QueryRouter()
        self.pipeline = RAGPipeline()
        self._crew: Optional[ResearchCrew] = None

    def _get_crew(self) -> ResearchCrew:
        """Get or create the research crew."""
        if self._crew is None:
            self._crew = ResearchCrew(
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                router=self.router,
                pipeline=self.pipeline,
                verbose=self.verbose,
                human_in_the_loop=self.human_in_the_loop,
            )
        return self._crew

    def query(self, question: str) -> ResearchResult:
        """Process a research query through the multi-agent system."""
        logger.info(f"Processing query: {question}")
        crew = self._get_crew()
        return crew.research(question)

    def index_document(self, file_path: str, chunking_strategy: str = "fixed") -> int:
        """Index a document for RAG retrieval."""
        logger.info(f"Indexing document: {file_path}")
        return self.pipeline.index_document(file_path, chunking_strategy)

    def index_documents(
        self, file_paths: List[str], chunking_strategy: str = "fixed"
    ) -> int:
        """Index multiple documents."""
        total_chunks = 0
        for path in file_paths:
            chunks = self.index_document(path, chunking_strategy)
            total_chunks += chunks
            logger.info(f"Indexed {chunks} chunks from {path}")
        return total_chunks

    def clear_index(self):
        """Clear all indexed documents."""
        self.pipeline.clear_index()
        logger.info("Index cleared")

    def get_indexed_count(self) -> int:
        """Get number of indexed chunks."""
        return self.pipeline.get_indexed_count()

    def get_status(self) -> dict:
        """Get current status of the research assistant."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "indexed_chunks": self.get_indexed_count(),
            "verbose": self.verbose,
            "human_in_the_loop": self.human_in_the_loop,
        }


def create_research_assistant(
    llm_provider: str = "ollama",
    llm_model: str = "ministral-3:3b",
    verbose: bool = False,
    human_in_the_loop: bool = False,
) -> ResearchAssistant:
    """Factory function to create a ResearchAssistant instance."""
    return ResearchAssistant(
        llm_provider=llm_provider,
        llm_model=llm_model,
        verbose=verbose,
        human_in_the_loop=human_in_the_loop,
    )
