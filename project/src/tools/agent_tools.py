import logging
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.routing.router import QueryRouter
from src.routing.models import RoutingMode
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class QueryAnalysisInput(BaseModel):
    query: str = Field(description="The user query to analyze")


class QueryAnalysisTool(BaseTool):
    name: str = "query_analysis"
    description: str = (
        "Analyzes a query to determine its complexity, type, and whether RAG retrieval is needed. "
        "Returns complexity_score (0-100), question_type, and analysis details."
    )
    args_schema: Type[BaseModel] = QueryAnalysisInput
    router: Optional[QueryRouter] = None

    def __init__(self, router: Optional[QueryRouter] = None, **kwargs):
        super().__init__(**kwargs)
        self.router = router or QueryRouter(mode=RoutingMode.BALANCED)

    def _run(self, query: str) -> str:
        try:
            decision, analysis = self.router.route(query)
            needs_rag = analysis.complexity_score >= 30 or analysis.question_type in ["analysis", "reasoning"]
            result = (
                f"Query Analysis Result:\n"
                f"- Complexity Score: {analysis.complexity_score}/100\n"
                f"- Question Type: {analysis.question_type}\n"
                f"- Has Complex Keywords: {analysis.has_complex_keywords}\n"
                f"- Quality Needed: {analysis.estimated_quality_needed}\n"
                f"- Token Count: {analysis.token_count}\n"
                f"- Needs RAG: {needs_rag}\n"
                f"- Recommended Model: {decision.provider}/{decision.model_name}\n"
                f"- Routing Reason: {decision.reason}"
            )
            return result
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return f"Error analyzing query: {str(e)}"


class RAGSearchInput(BaseModel):
    query: str = Field(description="The search query")
    top_k: int = Field(default=5, description="Number of results to retrieve", ge=1, le=20)


class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = (
        "Searches through indexed documents to find relevant information. "
        "Returns context chunks with source citations."
    )
    args_schema: Type[BaseModel] = RAGSearchInput
    pipeline: Optional[RAGPipeline] = None

    def __init__(self, pipeline: Optional[RAGPipeline] = None, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline or RAGPipeline()

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            indexed_count = self.pipeline.get_indexed_count()
            if indexed_count == 0:
                return "No documents indexed. RAG search unavailable."

            context = self.pipeline.query(query, top_k=top_k)
            if not context.results:
                return f"No relevant results found for query: {query}"

            formatted_results = []
            for result in context.results:
                source_info = f"[Source {result.rank}: {result.chunk.metadata.filename}, chunk {result.chunk.chunk_index}]"
                formatted_results.append(f"{source_info}\nScore: {result.score:.4f}\n{result.chunk.text}\n")

            output = (
                f"Retrieved {len(context.results)} results in {context.retrieval_time_ms:.2f}ms\n\n"
                + "\n---\n".join(formatted_results)
            )
            return output
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"Error during RAG search: {str(e)}"

    def get_indexed_count(self) -> int:
        if self.pipeline:
            return self.pipeline.get_indexed_count()
        return 0
