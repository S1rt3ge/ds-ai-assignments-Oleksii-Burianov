"""Prompt manager for creating and managing prompts with different strategies."""

from typing import Optional

from src.llm.models import Message, MessageRole
from src.prompts.strategies import STRATEGY_TEMPLATES, PromptStrategy


class PromptManager:
    """Manages prompt creation with different strategies."""

    def __init__(self, strategy: PromptStrategy = PromptStrategy.SYSTEM_ROLE):
        """
        Initialize the prompt manager.

        Args:
            strategy: The prompting strategy to use
        """
        self.strategy = strategy
        self.template = STRATEGY_TEMPLATES[strategy]

    def create_prompt(
        self,
        user_query: str,
        custom_system_prompt: Optional[str] = None,
    ) -> list[Message]:
        """
        Create a list of messages based on the selected strategy.

        Args:
            user_query: The user's query or question
            custom_system_prompt: Optional custom system prompt to override default

        Returns:
            List of Message objects ready for LLM consumption
        """
        messages = []

        # Add system message
        system_content = custom_system_prompt if custom_system_prompt else self.template["system"]
        messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

        # Add few-shot examples if using that strategy
        if self.strategy == PromptStrategy.FEW_SHOT and "examples" in self.template:
            for example in self.template["examples"]:
                messages.append(Message(role=MessageRole.USER, content=example["user"]))
                messages.append(Message(role=MessageRole.ASSISTANT, content=example["assistant"]))

        # Add user query with strategy-specific template
        user_content = self.template["user_template"].format(query=user_query)
        messages.append(Message(role=MessageRole.USER, content=user_content))

        return messages

    def get_strategy_description(self) -> str:
        """
        Get a description of the current strategy.

        Returns:
            Description string
        """
        return self.template.get("description", "")

    def set_strategy(self, strategy: PromptStrategy) -> None:
        """
        Change the prompting strategy.

        Args:
            strategy: New strategy to use
        """
        self.strategy = strategy
        self.template = STRATEGY_TEMPLATES[strategy]

    @staticmethod
    def list_strategies() -> list[dict[str, str]]:
        """
        List all available strategies with descriptions.

        Returns:
            List of dicts with 'name' and 'description' keys
        """
        return [
            {
                "name": strategy.value,
                "display_name": strategy.value.replace("_", " ").title(),
                "description": STRATEGY_TEMPLATES[strategy].get("description", ""),
            }
            for strategy in PromptStrategy
        ]
