import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --------------------------------------------------
# Load .env file
# --------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("🤖 My AI Chatbot")
st.write("Ask anything and get an AI-powered answer.")

# --------------------------------------------------
# Check Hugging Face Token
# --------------------------------------------------
if not HF_TOKEN:
    st.error("❌ Hugging Face token is missing.")
    st.info("Please create a .env file and add your HF_TOKEN.")
    st.stop()

# --------------------------------------------------
# Hugging Face Inference Client
# --------------------------------------------------
client = InferenceClient(
    provider="groq",
    api_key=HF_TOKEN
)

# --------------------------------------------------
# Chat History
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --------------------------------------------------
# User Input
# --------------------------------------------------
question = st.chat_input("Type your question here...")

if question:

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # AI response
    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                response = client.chat_completion(

                    model="openai/gpt-oss-20b",

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful, friendly and intelligent "
                                "AI assistant. Give clear and simple answers."
                            )
                        }
                    ] + st.session_state.messages,

                    max_tokens=300,
                    temperature=0.7
                )

                # Get answer
                answer = response.choices[0].message.content

                # Display answer
                st.write(answer)

                # Save answer
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error("❌ Unable to generate a response.")
                st.code(str(e))