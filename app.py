import streamlit as st
from chatbot import Chatbot

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource
def load_bot():
    return Chatbot()

st.title("🤖 AI Chatbot")
st.write("Ask me about Python, NLP, TensorFlow, or start a conversation.")

try:
    bot = load_bot()
except Exception as error:
    st.error("The chatbot model is not ready yet. In the VS Code terminal, run `python train.py`, then restart this app.")
    st.code(str(error))
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your AI chatbot. How can I help you?"}
    ]

with st.sidebar:
    st.subheader("About")
    st.write("This educational chatbot uses NLTK stemming and a TensorFlow neural network to classify message intents.")
    if st.button("Clear chat"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Chat cleared. What would you like to ask?"}
        ]
        st.rerun()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    answer = bot.get_response(prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
