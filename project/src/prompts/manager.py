from typing import Optional

from src.llm.models import Message, MessageRole
from src.prompts.strategies import STRATEGY_TEMPLATES, PromptStrategy


class PromptManager:
    def __init__(self, strategy: PromptStrategy = PromptStrategy.SYSTEM_ROLE):
        self.strategy = strategy
        self.template = STRATEGY_TEMPLATES[strategy]

    def create_prompt(
        self,
        user_query: str,
        custom_system_prompt: Optional[str] = None,
    ) -> list[Message]:
        messages = []

        system_content = custom_system_prompt if custom_system_prompt else self.template["system"]
        messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

        if self.strategy == PromptStrategy.FEW_SHOT and "examples" in self.template:
            for example in self.template["examples"]:
                messages.append(Message(role=MessageRole.USER, content=example["user"]))
                messages.append(Message(role=MessageRole.ASSISTANT, content=example["assistant"]))

        user_content = self.template["user_template"].format(query=user_query)
        messages.append(Message(role=MessageRole.USER, content=user_content))

        return messages

    def create_rag_prompt(self, user_query: str, context: str) -> list[Message]:
        messages = []

        system_content = (
            "You are a helpful research assistant. "
            "Use the provided context to answer questions accurately. "
            "Include inline citations referencing the source documents. "
            "If the context doesn't contain relevant information, say so."
        )
        messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

        user_content = f"""Use the following context to answer the question. Include inline citations.

Context:
{context}

Question: {user_query}

Answer with citations:"""

        messages.append(Message(role=MessageRole.USER, content=user_content))

        return messages

    def get_strategy_description(self) -> str:
        return self.template.get("description", "")

    def set_strategy(self, strategy: PromptStrategy) -> None:
        self.strategy = strategy
        self.template = STRATEGY_TEMPLATES[strategy]

    @staticmethod
    def list_strategies() -> list[dict[str, str]]:
        return [
            {
                "name": strategy.value,
                "display_name": strategy.value.replace("_", " ").title(),
                "description": STRATEGY_TEMPLATES[strategy].get("description", ""),
            }
            for strategy in PromptStrategy
        ]
