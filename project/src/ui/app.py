import streamlit as st

from src.llm import Message, MessageRole, get_llm_client
from src.prompts import PromptManager, PromptStrategy
from src.routing import QueryRouter, RoutingMode
from src.routing.strategies import MODEL_REGISTRY

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_routing_decision" not in st.session_state:
    st.session_state.last_routing_decision = None
if "total_cost_savings" not in st.session_state:
    st.session_state.total_cost_savings = 0.0

ALL_MODELS = {
    "Auto Route (Recommended)": "auto",
    "--- Local Models ---": "separator",
    "Ollama: gemma3:1b": ("ollama", "gemma3:1b"),
    "Ollama: deepseek-r1:1.5b": ("ollama", "deepseek-r1:1.5b"),
    "Ollama: ministral-3:3b": ("ollama", "ministral-3:3b"),
    "--- Cloud Models ---": "separator",
    "Cerebras: llama-3.3-70b": ("cerebras", "llama-3.3-70b"),
    "Mistral: medium-latest": ("mistral", "mistral-medium-latest"),
    "OpenRouter: gpt-4o": ("openrouter", "openai/gpt-4o"),
}


def main():
    st.title("Research Assistant")
    st.markdown("Multi-Agent Research System with Intelligent Routing & Prompting")

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Model Selection")

        model_options = list(ALL_MODELS.keys())
        selected_model_display = st.selectbox(
            "Choose Model",
            options=model_options,
            help="Select Auto Route for intelligent model selection or choose a specific model",
            index=0,
        )

        is_auto_route = ALL_MODELS[selected_model_display] == "auto"

        routing_mode = RoutingMode.BALANCED
        if is_auto_route:
            st.subheader("Routing Mode")
            mode_options = {
                "Balanced (Recommended)": RoutingMode.BALANCED,
                "Cost-Optimized": RoutingMode.COST_OPTIMIZED,
                "Quality-Optimized": RoutingMode.QUALITY_OPTIMIZED,
                "Ultra-Fast": RoutingMode.ULTRA_FAST,
            }

            selected_mode_display = st.selectbox(
                "Optimization Mode",
                options=list(mode_options.keys()),
                help="Choose how to balance cost, speed, and quality",
            )
            routing_mode = mode_options[selected_mode_display]

            router = QueryRouter(mode=routing_mode)
            st.info(router.explain_mode(routing_mode))

        st.subheader("Prompting Strategy")
        strategies = PromptManager.list_strategies()
        strategy_options = {s["display_name"]: s["name"] for s in strategies}

        selected_strategy_display = st.selectbox(
            "Choose Strategy",
            options=list(strategy_options.keys()),
            help="Select the prompting strategy to guide the model's responses",
        )
        selected_strategy = PromptStrategy(strategy_options[selected_strategy_display])

        current_strategy_info = next(s for s in strategies if s["name"] == selected_strategy.value)
        st.caption(current_strategy_info["description"])

        with st.expander("Advanced Options"):
            st.markdown("**Temperature** - Controls randomness in responses")
            st.caption("Lower (0.0) = More focused and deterministic")
            st.caption("Higher (2.0) = More creative and varied")
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, label_visibility="collapsed"
            )

            st.markdown("**Max Tokens** - Maximum length of response")
            st.caption("Limits how many tokens (words) the model can generate")
            max_tokens = st.number_input(
                "Max Tokens", min_value=100, max_value=4000, value=1000, step=100, label_visibility="collapsed"
            )

            st.markdown("**Streaming** - Real-time token generation")
            st.caption("Show response as it's being generated word-by-word")
            use_streaming = st.checkbox("Enable Streaming", value=False)

        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.session_state.last_routing_decision = None
            st.session_state.total_cost_savings = 0.0
            st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Research Question")

        user_query = st.text_area(
            "Your Query",
            placeholder="E.g., What are the main applications of deep learning in healthcare?",
            height=120,
            label_visibility="collapsed",
        )

        if st.button("Generate Response", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                if is_auto_route:
                    router = QueryRouter(mode=routing_mode)
                    decision, analysis = router.route(user_query, max_output_tokens=max_tokens)
                    st.session_state.last_routing_decision = decision

                    provider = decision.provider
                    model_name = decision.model_name

                    spinner_text = f"Auto-routing selected {provider}/{model_name}..."
                else:
                    provider, model_name = ALL_MODELS[selected_model_display]
                    st.session_state.last_routing_decision = None
                    spinner_text = f"Generating response with {provider}/{model_name}..."

                with st.spinner(spinner_text):
                    try:
                        prompt_manager = PromptManager(strategy=selected_strategy)
                        messages = prompt_manager.create_prompt(user_query)
                        client = get_llm_client(provider, model_name=model_name)

                        if use_streaming:
                            response_placeholder = st.empty()
                            full_response = ""

                            for chunk in client.stream(messages, temperature=temperature, max_tokens=max_tokens):
                                full_response += chunk.content
                                response_placeholder.markdown(full_response)

                                if chunk.is_final and chunk.metadata:
                                    st.session_state.last_response = chunk.metadata
                        else:
                            response = client.generate(messages, temperature=temperature, max_tokens=max_tokens)
                            st.session_state.last_response = response.metadata

                            st.markdown("### Response")
                            st.markdown(response.content)

                        if is_auto_route and st.session_state.last_response:
                            metadata = st.session_state.last_response

                            most_expensive = max(
                                MODEL_REGISTRY.values(),
                                key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
                            )
                            expensive_cost = most_expensive.estimate_cost(
                                metadata.input_tokens,
                                metadata.output_tokens
                            )
                            actual_cost = metadata.cost
                            savings = expensive_cost - actual_cost

                            if savings > 0:
                                st.session_state.total_cost_savings += savings

                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": user_query,
                                "model": f"{provider}/{model_name}",
                                "strategy": selected_strategy.value,
                                "auto_routed": is_auto_route,
                            }
                        )

                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"Error generating response: {error_msg}")

                        if provider == "ollama":
                            st.info(
                                f"Make sure Ollama is running and the model is pulled:\n```bash\nollama pull {model_name}\n```"
                            )
                        elif "402" in error_msg or "credits" in error_msg.lower():
                            st.warning(
                                f"**{provider.upper()} Credits Exhausted**\n\n"
                                f"Your free credits have been used up. Options:\n"
                                f"1. Use Auto-routing (will select cheaper models)\n"
                                f"2. Reduce max_tokens in Advanced Options\n"
                                f"3. Use Ollama (free local models)\n"
                                f"4. Use Cerebras or Mistral (generous free tiers)\n"
                                f"5. Add credits at https://openrouter.ai/settings/credits"
                            )
                        else:
                            st.info(f"Check that your {provider.upper()}_API_KEY is set in the .env file")

    with col2:
        if st.session_state.last_routing_decision:
            st.subheader("Routing Decision")
            decision = st.session_state.last_routing_decision

            st.success(f"**Selected:** {decision.provider}/{decision.model_name}")
            st.write(f"**Reason:** {decision.reason}")

            with st.expander("Decision Details"):
                st.write("**Factors:**")
                for key, value in decision.factors.items():
                    formatted_key = key.replace("_", " ").title()
                    st.write(f"• {formatted_key}: {value}")

                st.write("")
                st.write(f"**Complexity Score:** {decision.complexity_score}/100")
                st.metric("Estimated Cost", f"${decision.estimated_cost:.4f}")
                st.metric("Estimated Latency", f"{decision.estimated_latency_ms:.0f}ms")

        st.subheader("Response Metadata")

        if st.session_state.last_response:
            metadata = st.session_state.last_response

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Input Tokens", metadata.input_tokens)
                st.metric("Output Tokens", metadata.output_tokens)
            with col_b:
                st.metric("Total Tokens", metadata.total_tokens)
                st.metric("Cost", f"${metadata.cost:.4f}")

            st.metric("Latency", f"{metadata.latency_ms:.0f} ms")
            st.caption(f"Model: {metadata.model_name}")
            st.caption(f"Timestamp: {metadata.timestamp.strftime('%H:%M:%S')}")
        else:
            st.info("Generate a response to see metadata")

        if st.session_state.total_cost_savings > 0:
            st.subheader("Cost Savings")
            st.success(f"Total saved: ${st.session_state.total_cost_savings:.4f}")
            most_expensive = max(
                MODEL_REGISTRY.values(),
                key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
            )
            st.caption(f"Compared to always using {most_expensive.provider}/{most_expensive.model_name}")

    if st.session_state.messages:
        st.markdown("---")
        st.subheader("Conversation History")

        for i, msg in enumerate(reversed(st.session_state.messages[-5:])):
            auto_badge = "[AUTO] " if msg.get("auto_routed", False) else ""
            with st.expander(f"{auto_badge}Query {len(st.session_state.messages) - i}: {msg['content'][:50]}..."):
                st.write(f"**Model:** {msg['model']}")
                st.write(f"**Strategy:** {msg['strategy']}")
                if msg.get("auto_routed"):
                    st.write("**Routing:** Auto-selected")
                st.write(f"**Query:** {msg['content']}")


if __name__ == "__main__":
    main()
