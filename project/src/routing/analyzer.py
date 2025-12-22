import re

from src.routing.models import QueryAnalysis, QuestionType


class QueryAnalyzer:
    FACTUAL_KEYWORDS = [
        "what is",
        "define",
        "who is",
        "when did",
        "where is",
        "list",
        "name",
        "tell me about",
    ]

    REASONING_KEYWORDS = [
        "why",
        "how does",
        "explain",
        "describe",
        "what causes",
        "what makes",
        "reason",
    ]

    ANALYSIS_KEYWORDS = [
        "analyze",
        "compare",
        "contrast",
        "evaluate",
        "assess",
        "examine",
        "investigate",
        "differences between",
        "similarities between",
    ]

    CREATIVE_KEYWORDS = [
        "write",
        "create",
        "generate",
        "compose",
        "imagine",
        "design",
        "build",
        "make a story",
        "poem",
    ]

    COMPLEX_KEYWORDS = [
        "theoretical",
        "implications",
        "comprehensive",
        "detailed",
        "in-depth",
        "thoroughly",
        "extensively",
        "critically",
        "deeply",
        "advanced",
        "complex",
        "sophisticated",
    ]

    def __init__(self):
        pass

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        word_count = len(query.split())
        token_count = int(word_count * 0.75)
        question_type = self._determine_question_type(query_lower)
        has_complex_keywords = self._has_complex_keywords(query_lower)

        complexity_score = self._calculate_complexity_score(
            token_count=token_count,
            question_type=question_type,
            has_complex_keywords=has_complex_keywords,
            query_lower=query_lower,
        )
        estimated_quality_needed = self._estimate_quality_needed(complexity_score)

        return QueryAnalysis(
            query=query,
            token_count=token_count,
            question_type=question_type,
            has_complex_keywords=has_complex_keywords,
            complexity_score=complexity_score,
            estimated_quality_needed=estimated_quality_needed,
        )

    def _determine_question_type(self, query_lower: str) -> QuestionType:
        if any(keyword in query_lower for keyword in self.CREATIVE_KEYWORDS):
            return QuestionType.CREATIVE

        if any(keyword in query_lower for keyword in self.ANALYSIS_KEYWORDS):
            return QuestionType.ANALYSIS

        if any(keyword in query_lower for keyword in self.REASONING_KEYWORDS):
            return QuestionType.REASONING

        if any(keyword in query_lower for keyword in self.FACTUAL_KEYWORDS):
            return QuestionType.FACTUAL

        return QuestionType.UNKNOWN

    def _has_complex_keywords(self, query_lower: str) -> bool:
        return any(keyword in query_lower for keyword in self.COMPLEX_KEYWORDS)

    def _calculate_complexity_score(
        self,
        token_count: int,
        question_type: QuestionType,
        has_complex_keywords: bool,
        query_lower: str,
    ) -> int:
        score = 0

        if token_count < 50:
            score += 0
        elif token_count < 100:
            score += 10
        elif token_count < 200:
            score += 20
        else:
            score += 40

        type_scores = {
            QuestionType.FACTUAL: 0,
            QuestionType.REASONING: 15,
            QuestionType.ANALYSIS: 25,
            QuestionType.CREATIVE: 30,
            QuestionType.UNKNOWN: 10,
        }
        score += type_scores.get(question_type, 10)

        if has_complex_keywords:
            score += 20

        sentence_count = len(re.split(r"[.!?]+", query_lower))
        if sentence_count > 2:
            score += 10

        return min(100, max(0, score))

    def _estimate_quality_needed(self, complexity_score: int) -> str:
        if complexity_score < 30:
            return "low"
        elif complexity_score < 70:
            return "medium"
        else:
            return "high"
