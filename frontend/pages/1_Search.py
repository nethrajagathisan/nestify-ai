import streamlit as st
import requests

st.title("🔍 Property Search")
st.markdown("Find your ideal property with natural language and smart filters.")
st.divider()

API_BASE = "http://localhost:8000"

CITIES = ["All", "Bangalore", "Chennai", "Hyderabad", "Mumbai", "Pune", "Delhi", "Coimbatore"]
BHK_OPTIONS = ["Any", "1", "2", "3", "4"]

# Layout: query on top, filters in sidebar (or left column per request)
col_left, col_right = st.columns([2, 1])

with col_left:
    query = st.text_input(
        "What are you looking for?",
        placeholder="2BHK apartment near metro under ₹80L",
    )

with col_right:
    with st.container(border=True):
        st.markdown("**Filters**")
        city_filter = st.selectbox("City", CITIES)
        bhk_filter = st.selectbox("BHK", BHK_OPTIONS)
        min_price, max_price = st.slider(
            "Price Range (₹ Lakhs)",
            min_value=10,
            max_value=500,
            value=(20, 150),
            step=5,
        )

st.divider()

if st.button("🔎 Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        payload = {
            "query": query,
            "top_k": 10,
        }
        if city_filter != "All":
            payload["city"] = city_filter
        if bhk_filter != "Any":
            payload["bhk"] = int(bhk_filter)
        payload["min_price_lakhs"] = float(min_price)
        payload["max_price_lakhs"] = float(max_price)

        with st.spinner("Searching properties..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/properties/search",
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Make sure the API server is running on localhost:8000.")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        results = data.get("results", [])
        total_found = data.get("total_found", len(results))
        llm_summary = data.get("llm_summary", "")

        st.success(f"Found {total_found} properties")

        if llm_summary:
            with st.expander("🤖 AI Summary", expanded=True):
                st.markdown(llm_summary)

        if not results:
            st.info("No properties matched your criteria. Try adjusting the filters.")
        else:
            for prop in results:
                with st.container(border=True):
                    price = prop.get("price_lakhs", 0)
                    price_color = "red" if price > 200 else "green" if price < 50 else "inherit"
                    price_display = f"<span style='color:{price_color}; font-size:1.6rem; font-weight:bold;'>₹{price:,.2f} L</span>"

                    st.markdown(f"**{prop.get('title', 'Untitled')}**")
                    st.markdown(price_display, unsafe_allow_html=True)

                    meta_cols = st.columns(4)
                    with meta_cols[0]:
                        st.metric("BHK", prop.get("bhk", "N/A"))
                    with meta_cols[1]:
                        st.metric("Area", f"{prop.get('area_sqft', 0):,} sqft")
                    with meta_cols[2]:
                        st.metric("Type", prop.get("property_type", "N/A").title())
                    with meta_cols[3]:
                        year = prop.get("year_built")
                        st.metric("Year Built", year if year else "N/A")

                    amenities = prop.get("amenities", [])
                    if amenities:
                        st.markdown(
                            "**Amenities:** " + ", ".join(f"`{a}`" for a in amenities)
                        )

                    desc = prop.get("description", "")
                    if desc:
                        st.markdown(f"*{desc}*", help="Property description")

        st.divider()
        st.caption("Results powered by AI retrieval + semantic search.")
