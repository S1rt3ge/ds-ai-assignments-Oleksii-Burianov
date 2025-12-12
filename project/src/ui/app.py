"""Streamlit UI for the Research Assistant system."""

import streamlit as st

from src.llm import Message, MessageRole, get_llm_client, list_available_models
from src.prompts import PromptManager, PromptStrategy

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


def main():
    """Main Streamlit application."""

    # Title and description
    st.title("Research Assistant")
    st.markdown("Multi-Agent Research System with Intelligent Prompting")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        st.subheader("Model Selection")
        available_models = list_available_models("ollama")
        selected_model = st.selectbox(
            "Choose Model",
            options=available_models,
            help="Select which Ollama model to use for generation",
        )

        # Strategy selection
        st.subheader("Prompting Strategy")
        strategies = PromptManager.list_strategies()
        strategy_options = {s["display_name"]: s["name"] for s in strategies}

        selected_strategy_display = st.selectbox(
            "Choose Strategy",
            options=list(strategy_options.keys()),
            help="Select the prompting strategy to guide the model's responses",
        )
        selected_strategy = PromptStrategy(strategy_options[selected_strategy_display])

        # Show strategy description
        current_strategy_info = next(s for s in strategies if s["name"] == selected_strategy.value)
        st.info(current_strategy_info['description'])

        # Advanced options with explanations
        with st.expander("Advanced Options"):
            st.markdown("**Temperature** - Controls randomness in responses")
            st.caption("Lower (0.0) = More focused and deterministic")
            st.caption("Higher (2.0) = More creative and varied")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                label_visibility="collapsed"
            )

            st.markdown("**Max Tokens** - Maximum length of response")
            st.caption("Limits how many tokens (words) the model can generate")
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                label_visibility="collapsed"
            )

            st.markdown("**Streaming** - Real-time token generation")
            st.caption("Show response as it's being generated word-by-word")
            use_streaming = st.checkbox("Enable Streaming", value=False)

        # Clear conversation
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Research Question")

        # User input
        user_query = st.text_area(
            "Your Query",
            placeholder="E.g., What are the main applications of deep learning in healthcare?",
            height=120,
            label_visibility="collapsed",
        )

        # Generate button
        if st.button("Generate Response", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner(f"Generating response with {selected_model}..."):
                    try:
                        # Create prompt manager and messages
                        prompt_manager = PromptManager(strategy=selected_strategy)
                        messages = prompt_manager.create_prompt(user_query)

                        # Initialize LLM client
                        client = get_llm_client("ollama", model_name=selected_model)

                        # Generate response
                        if use_streaming:
                            response_placeholder = st.empty()
                            full_response = ""

                            for chunk in client.stream(messages, temperature=temperature):
                                full_response += chunk.content
                                response_placeholder.markdown(full_response)

                                if chunk.is_final and chunk.metadata:
                                    st.session_state.last_response = chunk.metadata
                        else:
                            response = client.generate(messages, temperature=temperature)
                            st.session_state.last_response = response.metadata

                            # Display response
                            st.markdown("### Response")
                            st.markdown(response.content)

                        # Store in conversation history
                        st.session_state.messages.append(
                            {"role": "user", "content": user_query, "model": selected_model, "strategy": selected_strategy.value}
                        )

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info(f"Make sure Ollama is running and the model is pulled:\n```bash\nollama pull {selected_model}\n```")

    with col2:
        st.subheader("Response Metadata")

        if st.session_state.last_response:
            metadata = st.session_state.last_response

            # Display metrics
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

        # Model comparison info
        st.subheader("Model Info")
        model_info = {
            "gemma3:1b": {"size": "1B params", "speed": "Fast", "quality": "Basic"},
            "deepseek-r1:1.5b": {"size": "1.5B params", "speed": "Medium", "quality": "Good"},
            "ministral-3:3b": {"size": "3B params", "speed": "Slow", "quality": "High"},
        }

        if selected_model in model_info:
            info = model_info[selected_model]
            st.write(f"**Size:** {info['size']}")
            st.write(f"**Speed:** {info['speed']}")
            st.write(f"**Quality:** {info['quality']}")

    # Conversation history
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("Conversation History")

        for i, msg in enumerate(reversed(st.session_state.messages[-5:])):
            with st.expander(f"Query {len(st.session_state.messages) - i}: {msg['content'][:50]}..."):
                st.write(f"**Model:** {msg['model']}")
                st.write(f"**Strategy:** {msg['strategy']}")
                st.write(f"**Query:** {msg['content']}")


if __name__ == "__main__":
    main()
