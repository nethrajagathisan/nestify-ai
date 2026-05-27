import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Property Comparison",
    page_icon="🔍",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

st.title("Property Comparison")

# Two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property 1")
    prop1_input = st.text_input(
        "Enter property title or description",
        placeholder="e.g., 3BHK apartment in Whitefield, Bangalore",
        key="prop1"
    )

with col2:
    st.subheader("Property 2")
    prop2_input = st.text_input(
        "Enter property title or description",
        placeholder="e.g., 2BHK villa in HSR Layout, Bangalore",
        key="prop2"
    )

# Find & Compare button
if st.button("Find & Compare", type="primary", use_container_width=True):
    if not prop1_input or not prop2_input:
        st.warning("Please enter descriptions for both properties")
    else:
        with st.spinner("Searching for properties..."):
            try:
                # Search for Property 1
                response1 = requests.post(
                    f"{API_BASE}/properties/search",
                    json={
                        "query": prop1_input,
                        "top_k": 1
                    },
                    timeout=30
                )
                response1.raise_for_status()
                data1 = response1.json()
                
                # Search for Property 2
                response2 = requests.post(
                    f"{API_BASE}/properties/search",
                    json={
                        "query": prop2_input,
                        "top_k": 1
                    },
                    timeout=30
                )
                response2.raise_for_status()
                data2 = response2.json()
                
                # Extract property IDs
                if not data1["results"] or not data2["results"]:
                    st.error("Could not find matching properties. Please try different descriptions.")
                else:
                    prop1 = data1["results"][0]
                    prop2 = data2["results"][0]
                    prop1_id = prop1.id
                    prop2_id = prop2.id
                    
                    # Compare properties
                    compare_response = requests.post(
                        f"{API_BASE}/properties/compare",
                        json={"property_ids": [prop1_id, prop2_id]},
                        timeout=30
                    )
                    compare_response.raise_for_status()
                    comparison = compare_response.json()
                    
                    # Display results in two columns
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.subheader("Property A")
                        st.markdown(f"**Title:** {prop1.title}")
                        st.markdown(f"**City:** {prop1.city.value}")
                        st.markdown(f"**Locality:** {prop1.locality}")
                        st.markdown(f"**BHK:** {prop1.bhk.value}")
                        st.markdown(f"**Price:** ₹{prop1.price_lakhs} Lakhs")
                        st.markdown(f"**Area:** {prop1.area_sqft} sqft")
                        st.markdown(f"**Type:** {prop1.property_type.value}")
                        if prop1.amenities:
                            st.markdown(f"**Amenities:** {', '.join(prop1.amenities)}")
                        st.markdown(f"**Description:** {prop1.description}")
                    
                    with col_b:
                        st.subheader("Property B")
                        st.markdown(f"**Title:** {prop2.title}")
                        st.markdown(f"**City:** {prop2.city.value}")
                        st.markdown(f"**Locality:** {prop2.locality}")
                        st.markdown(f"**BHK:** {prop2.bhk.value}")
                        st.markdown(f"**Price:** ₹{prop2.price_lakhs} Lakhs")
                        st.markdown(f"**Area:** {prop2.area_sqft} sqft")
                        st.markdown(f"**Type:** {prop2.property_type.value}")
                        if prop2.amenities:
                            st.markdown(f"**Amenities:** {', '.join(prop2.amenities)}")
                        st.markdown(f"**Description:** {prop2.description}")
                    
                    # Display metrics
                    st.divider()
                    st.subheader("Comparison Metrics")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric(
                            "Price per sqft (Property A)",
                            f"₹{comparison['metrics_a']['price_per_sqft']}/sqft"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Price per sqft (Property B)",
                            f"₹{comparison['metrics_b']['price_per_sqft']}/sqft"
                        )
                    
                    metric_col3, metric_col4 = st.columns(2)
                    
                    with metric_col3:
                        age_a = comparison['metrics_a']['age_years']
                        st.metric(
                            "Building age (Property A)",
                            f"{age_a} years" if age_a else "N/A"
                        )
                    
                    with metric_col4:
                        age_b = comparison['metrics_b']['age_years']
                        st.metric(
                            "Building age (Property B)",
                            f"{age_b} years" if age_b else "N/A"
                        )
                    
                    # Display LLM comparison
                    st.divider()
                    st.subheader("AI Comparison")
                    st.info(comparison['comparison'])
                    
                    # Show more details button
                    if st.button("Show more details", key="show_details"):
                        st.json(comparison)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")
