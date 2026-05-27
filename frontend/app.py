import streamlit as st

st.set_page_config(
    page_title="Real Estate Copilot India",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Nestify-ai")
st.markdown("Find properties, compare listings, and get legal advice—all in one place.")
st.divider()

search_page = st.Page("pages/1_Search.py", title="🔍 Property Search", icon="🔍")
compare_page = st.Page("pages/2_Compare.py", title="⚖️ Compare", icon="⚖️")
legal_page = st.Page("pages/3_Legal.py", title="📜 Legal FAQ", icon="📜")
chat_page = st.Page("pages/4_Chat.py", title="💬 Ask Copilot", icon="💬")

pg = st.navigation([search_page, compare_page, legal_page, chat_page])

with st.sidebar:
    st.header("Supported Cities")
    st.markdown(
        """
        - Bangalore
        - Chennai
        - Hyderabad
        - Mumbai
        - Pune
        - Delhi
        - Coimbatore
        """
    )
    st.divider()
    st.caption("Powered by AI · Data refreshed periodically")

pg.run()
