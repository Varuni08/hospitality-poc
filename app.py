import streamlit as st
from orchestrator import MultiAgentOrchestrator

st.set_page_config(
    page_title="Saffron Table Bistro",
    layout="centered"
)

st.title("Saffron Table Bistro")
st.caption("Your personal restaurant assistant")

if "bot" not in st.session_state:
    st.session_state.bot     = MultiAgentOrchestrator()
if "session" not in st.session_state:
    st.session_state.session = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Welcome to Saffron Table Bistro! How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.handle(prompt, st.session_state.session)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
