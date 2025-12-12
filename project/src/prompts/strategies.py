"""Prompt strategies for different use cases."""

from enum import Enum


class PromptStrategy(str, Enum):
    """Available prompting strategies."""

    SYSTEM_ROLE = "system_role"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STRUCTURED_OUTPUT = "structured_output"


# Base system prompts for Research Assistant domain
RESEARCH_ASSISTANT_BASE = """You are an expert research assistant specializing in academic literature analysis, summarization, and report generation. Your role is to help users with research tasks efficiently and accurately."""

# Strategy-specific templates
STRATEGY_TEMPLATES = {
    PromptStrategy.SYSTEM_ROLE: {
        "system": RESEARCH_ASSISTANT_BASE,
        "user_template": "{query}",
        "description": "Clear role definition with direct instructions",
    },
    PromptStrategy.FEW_SHOT: {
        "system": RESEARCH_ASSISTANT_BASE
        + "\n\nHere are some examples of how to respond to research queries:",
        "examples": [
            {
                "user": "Summarize the key findings of a paper on transformer architectures.",
                "assistant": "Key findings from transformer architecture research:\n\n1. Self-attention mechanism enables parallel processing of sequences\n2. Eliminates need for recurrent connections, improving training efficiency\n3. Achieves state-of-the-art results on machine translation tasks\n4. Scales effectively with increased model size and data\n\nThe transformer's attention mechanism allows it to capture long-range dependencies more effectively than RNNs or LSTMs.",
            },
            {
                "user": "What are the main research gaps in climate change modeling?",
                "assistant": "Main research gaps in climate change modeling:\n\n1. **Regional Predictions**: Global models struggle with localized climate impacts\n2. **Feedback Loops**: Incomplete understanding of ecosystem feedback mechanisms\n3. **Tipping Points**: Difficulty predicting threshold events and cascading effects\n4. **Socioeconomic Integration**: Limited coupling of human behavior with physical models\n5. **Computational Limitations**: Trade-offs between resolution and simulation time\n\nAddressing these gaps requires interdisciplinary collaboration and enhanced computational resources.",
            },
        ],
        "user_template": "{query}",
        "description": "Learning from examples before answering",
    },
    PromptStrategy.CHAIN_OF_THOUGHT: {
        "system": RESEARCH_ASSISTANT_BASE
        + "\n\nWhen answering research questions, think step-by-step and show your reasoning process. Break down complex problems into smaller parts.",
        "user_template": "{query}\n\nLet's approach this step-by-step:",
        "description": "Step-by-step reasoning and explanation",
    },
    PromptStrategy.STRUCTURED_OUTPUT: {
        "system": RESEARCH_ASSISTANT_BASE
        + "\n\nProvide responses in a structured JSON format with clear sections.",
        "user_template": """{query}

Please provide your response in the following JSON format:
{{
    "summary": "Brief overview of the topic",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "details": "Detailed explanation",
    "sources": ["Relevant source or context 1", "Source 2"],
    "confidence": "high/medium/low"
}}""",
        "description": "JSON-formatted structured responses",
    },
}
