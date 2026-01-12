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

    def _run(self, query: str = None, **kwargs) -> str:
        if query is None:
            query = kwargs.get("properties", {}).get("query", "")
        if not query:
            return "Error: No query provided"
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
            return f"Error analyzing query: {str(e)}. Check if routing system is initialized."


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

    def _run(self, query: str = None, top_k: int = 5, **kwargs) -> str:
        if query is None:
            props = kwargs.get("properties", {})
            query = props.get("query", "")
            top_k = props.get("top_k", 5)
        if not query:
            return "Error: No query provided"
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
            return f"Error during RAG search: {str(e)}. Check if documents are indexed."

    def get_indexed_count(self) -> int:
        if self.pipeline:
            return self.pipeline.get_indexed_count()
        return 0


class SummarizerInput(BaseModel):
    text: str = Field(description="The text to summarize")
    max_sentences: int = Field(default=3, description="Maximum sentences in summary", ge=1, le=10)


class SummarizerTool(BaseTool):
    name: str = "summarizer"
    description: str = (
        "Summarizes long text into key points. "
        "Returns a concise summary with the main ideas."
    )
    args_schema: Type[BaseModel] = SummarizerInput
    llm_client: Optional[any] = None

    def __init__(self, llm_client: Optional[any] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = llm_client

    def _run(self, text: str = None, max_sentences: int = 3, **kwargs) -> str:
        if text is None:
            props = kwargs.get("properties", {})
            text = props.get("text", "")
            max_sentences = props.get("max_sentences", 3)
        if not text:
            return "Error: No text provided"
        if len(text) < 100:
            return f"Text too short to summarize. Original: {text}"
        try:
            sentences = text.replace("!", ".").replace("?", ".").split(".")
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
            if len(sentences) <= max_sentences:
                return f"Summary: {text}"
            key_sentences = sentences[:max_sentences]
            summary = ". ".join(key_sentences) + "."
            return (
                f"Summary ({max_sentences} key points):\n{summary}\n\n"
                f"Original length: {len(text)} chars, Summary length: {len(summary)} chars"
            )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error summarizing text: {str(e)}"


class FactCheckerInput(BaseModel):
    claim: str = Field(description="The claim or statement to verify")
    context: str = Field(default="", description="Optional context to check against")


class FactCheckerTool(BaseTool):
    name: str = "fact_checker"
    description: str = (
        "Verifies a claim against provided context or indexed documents. "
        "Returns verification result with supporting evidence."
    )
    args_schema: Type[BaseModel] = FactCheckerInput
    pipeline: Optional[RAGPipeline] = None

    def __init__(self, pipeline: Optional[RAGPipeline] = None, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def _run(self, claim: str = None, context: str = "", **kwargs) -> str:
        if claim is None:
            props = kwargs.get("properties", {})
            claim = props.get("claim", "")
            context = props.get("context", "")
        if not claim:
            return "Error: No claim provided"
        try:
            if not context and self.pipeline and self.pipeline.get_indexed_count() > 0:
                rag_context = self.pipeline.query(claim, top_k=3)
                if rag_context.results:
                    context = " ".join([r.chunk.text for r in rag_context.results])
            if not context:
                return (
                    f"Claim: {claim}\n"
                    f"Verification: UNVERIFIED - No context available to check against.\n"
                    f"Recommendation: Provide context or index relevant documents."
                )
            claim_lower = claim.lower()
            context_lower = context.lower()
            claim_words = set(claim_lower.split())
            context_words = set(context_lower.split())
            overlap = claim_words.intersection(context_words)
            overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0
            if overlap_ratio > 0.6:
                status = "LIKELY SUPPORTED"
                confidence = "high"
            elif overlap_ratio > 0.3:
                status = "PARTIALLY SUPPORTED"
                confidence = "medium"
            else:
                status = "NOT DIRECTLY SUPPORTED"
                confidence = "low"
            return (
                f"Claim: {claim}\n"
                f"Verification: {status}\n"
                f"Confidence: {confidence}\n"
                f"Evidence overlap: {overlap_ratio:.0%}\n"
                f"Matching terms: {', '.join(list(overlap)[:10])}"
            )
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            return f"Error checking fact: {str(e)}"
