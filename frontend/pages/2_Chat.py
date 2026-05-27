import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="💬 Conversational Search",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filters" not in st.session_state:
    st.session_state.filters = {}

# API base URL
API_BASE = "http://localhost:8000/api/v1"

# Sidebar - Filters
with st.sidebar:
    st.subheader("Refine your search")
    
    # City filter
    city_options = ["", "Bangalore", "Chennai", "Hyderabad", "Mumbai", "Pune", "Delhi", "Coimbatore"]
    selected_city = st.selectbox(
        "City",
        city_options,
        index=city_options.index(st.session_state.filters.get("city", "")) if st.session_state.filters.get("city") in city_options else 0,
        key="city_filter"
    )
    if selected_city:
        st.session_state.filters["city"] = selected_city
    elif "city" in st.session_state.filters:
        del st.session_state.filters["city"]
    
    # BHK filter
    bhk_options = ["", 1, 2, 3, 4]
    selected_bhk = st.selectbox(
        "BHK",
        bhk_options,
        index=bhk_options.index(st.session_state.filters.get("bhk", "")) if st.session_state.filters.get("bhk") in bhk_options else 0,
        key="bhk_filter"
    )
    if selected_bhk:
        st.session_state.filters["bhk"] = selected_bhk
    elif "bhk" in st.session_state.filters:
        del st.session_state.filters["bhk"]
    
    # Price range filter
    min_price, max_price = st.slider(
        "Price Range (Lakhs)",
        min_value=0,
        max_value=200,
        value=(
            st.session_state.filters.get("min_price_lakhs", 0),
            st.session_state.filters.get("max_price_lakhs", 200)
        ),
        key="price_filter"
    )
    if min_price > 0:
        st.session_state.filters["min_price_lakhs"] = min_price
    elif "min_price_lakhs" in st.session_state.filters:
        del st.session_state.filters["min_price_lakhs"]
    
    if max_price < 200:
        st.session_state.filters["max_price_lakhs"] = max_price
    elif "max_price_lakhs" in st.session_state.filters:
        del st.session_state.filters["max_price_lakhs"]
    
    st.divider()
    
    # Clear buttons
    if st.button("Clear filters", use_container_width=True):
        st.session_state.filters = {}
        st.rerun()
    
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area - Chat interface
st.title("💬 Conversational Search")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show intent badge for assistant messages
        if message["role"] == "assistant" and "intent" in message:
            st.caption(f"Intent: {message['intent']}")

# Chat input
if prompt := st.chat_input("Ask about properties, compare listings, or legal questions..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "query": prompt,
                        "history": st.session_state.messages[:-1],  # Exclude current message
                        "filters": st.session_state.filters
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                assistant_response = data["response"]
                intent = data["intent"]
                sources = data.get("sources", [])
                
                st.markdown(assistant_response)
                st.caption(f"Intent: {intent}")
                
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "intent": intent,
                    "sources": sources
                })
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "intent": "error"
                })
