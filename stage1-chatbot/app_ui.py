import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Stage 2: Web AI Chatbot")
st.caption("Powered by Groq & Qwen 27B")

# 2. Load API credentials
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing. Check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

# 3. Streamlit Session State (Persistent Web Memory)
# Unlike a terminal script that keeps running, Streamlit reruns the script 
# from top to bottom on every user interaction. We use st.session_state 
# so memory is not erased on each click or input.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a sharp, conversational, and direct AI assistant. Keep responses clear and engaging."
        }
    ]

# Sidebar: Controls
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# 4. Display Existing Chat Messages on Screen
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. Handle New User Input
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Store user message into session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Generate response from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=st.session_state.messages,
                    temperature=temperature
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)

                # Store assistant response into session state
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Error calling Groq API: {e}")