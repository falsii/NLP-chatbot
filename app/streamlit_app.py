import os
import sys
import streamlit as st


# Add project root and src folder to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


from chat import load_chatbot, predict_chatbot


st.set_page_config(
    page_title="Offline PyTorch NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)


@st.cache_resource
def load_resources():
    """
    Load model, dataset, FAQ retriever, and required metadata only once.
    Streamlit reruns the script on every interaction, so caching is important.
    """

    return load_chatbot()


def initialize_session_state():
    """
    Store chat history and settings in Streamlit session state.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your offline PyTorch chatbot. How can I help you today?",
                "metadata": None
            }
        ]

    if "show_debug" not in st.session_state:
        st.session_state.show_debug = True


def render_sidebar():
    """
    Sidebar content for portfolio explanation and controls.
    """

    st.sidebar.title("Project Info")

    st.sidebar.markdown(
        """
        **Offline NLP Chatbot**

        This chatbot uses:

        - PyTorch model trained from scratch
        - Bag-of-Words NLP preprocessing
        - Rule-based safety layer
        - TF-IDF FAQ retrieval
        - Confidence-based fallback

        No LLM, no API key, no pretrained model.
        """
    )

    st.sidebar.divider()

    st.session_state.show_debug = st.sidebar.checkbox(
        "Show debug details",
        value=st.session_state.show_debug
    )

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. How can I help you?",
                "metadata": None
            }
        ]
        st.rerun()

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **Try asking:**

        - how do i reset my password
        - what can you do
        - how much does the product cost
        - how do i contact support
        - are you a human
        - i need medical help
        """
    )


def render_chat_messages():
    """
    Display chat messages from session state.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            metadata = message.get("metadata")

            if (
                message["role"] == "assistant"
                and metadata
                and st.session_state.show_debug
            ):
                with st.expander("Debug details"):
                    st.write(f"**Source:** {metadata.get('source')}")
                    st.write(f"**Intent:** {metadata.get('intent')}")
                    st.write(f"**Confidence:** {metadata.get('confidence'):.2f}")

                    if metadata.get("faq_match_score") is not None:
                        st.write(f"**FAQ Score:** {metadata.get('faq_match_score'):.2f}")

                    if metadata.get("faq_question"):
                        st.write(f"**Matched FAQ:** {metadata.get('faq_question')}")


def main():
    initialize_session_state()

    st.title("Offline PyTorch NLP Chatbot")
    st.caption("Trained from scratch | No LLM | No API key | No pretrained model")

    render_sidebar()

    try:
        model, intents, all_words, tags, device, retriever = load_resources()
    except FileNotFoundError as error:
        st.error("Required file not found.")
        st.code(str(error))
        st.info("Make sure you have trained the model using: python src/train.py")
        return
    except Exception as error:
        st.error("Something went wrong while loading chatbot resources.")
        st.code(str(error))
        return

    render_chat_messages()

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "metadata": None
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        result = predict_chatbot(
            user_message=user_input,
            model=model,
            intents=intents,
            all_words=all_words,
            tags=tags,
            device=device,
            retriever=retriever
        )

        bot_response = result["response"]

        assistant_message = {
            "role": "assistant",
            "content": bot_response,
            "metadata": result
        }

        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(bot_response)

            if st.session_state.show_debug:
                with st.expander("Debug details"):
                    st.write(f"**Source:** {result.get('source')}")
                    st.write(f"**Intent:** {result.get('intent')}")
                    st.write(f"**Confidence:** {result.get('confidence'):.2f}")

                    if result.get("faq_match_score") is not None:
                        st.write(f"**FAQ Score:** {result.get('faq_match_score'):.2f}")

                    if result.get("faq_question"):
                        st.write(f"**Matched FAQ:** {result.get('faq_question')}")


if __name__ == "__main__":
    main()