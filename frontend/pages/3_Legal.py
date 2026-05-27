import streamlit as st
import requests

st.title("📜 Legal FAQ")
st.markdown("Ask questions about Indian property law, stamp duty, RERA, and more.")
st.divider()

API_BASE = "http://localhost:8000"

question = st.text_area(
    "Your legal question",
    placeholder="What is RERA and how does it protect home buyers?",
    height=100,
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Researching..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/faq/faq",
                    json={"question": question},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data.get("answer", "No answer returned."))
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"- {src}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Make sure the API server is running on localhost:8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.divider()
st.caption("Disclaimer: This is general information, not legal advice. Consult a qualified lawyer or RERA authority for your specific situation.")
