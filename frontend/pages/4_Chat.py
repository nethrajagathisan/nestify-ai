import streamlit as st
import requests

st.title("💬 Ask Copilot")
st.markdown("Have a free-form conversation with the Real Estate Copilot.")
st.divider()

API_BASE = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about Indian real estate...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/chat",
                    json={
                        "query": user_input,
                        "history": st.session_state.chat_history[:-1],
                        "filters": {},
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data.get("response", "I'm not sure how to answer that.")
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except requests.exceptions.ConnectionError:
                err = "Could not connect to the backend. Make sure the API server is running on localhost:8000."
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})
