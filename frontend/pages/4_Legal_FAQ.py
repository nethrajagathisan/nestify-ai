import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="⚖️ Legal FAQ & Regulations",
    page_icon="⚖️",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

st.title("⚖️ Legal FAQ & Regulations")

# Sidebar with tags
with st.sidebar:
    st.subheader("Quick Topics")
    tags = ["RERA", "Stamp Duty", "Home Loan Tax", "Registration", "NRI Guide"]
    for tag in tags:
        st.button(tag, use_container_width=True, key=f"tag_{tag}")

# Main area
st.subheader("Ask about Indian property law")

question = st.text_input(
    "Your question",
    placeholder="What is stamp duty in Karnataka?",
    key="legal_question"
)

if st.button("Ask", type="primary"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching legal documents..."):
            try:
                response = requests.post(
                    f"{API_BASE}/faq",
                    json={"question": question},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Display answer
                st.info(data["answer"])
                
                # Display sources as badges
                if data["sources"]:
                    st.subheader("Information sources")
                    for source in data["sources"]:
                        st.caption(f"📄 {source}")
                
                # Disclaimer
                st.warning("⚠️ This is general information. Consult a lawyer for specific advice.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")

# FAQ accordion with sample questions
st.divider()
st.subheader("Frequently Asked Questions")

FAQS = {
    "What is RERA?": "The Real Estate (Regulation and Development) Act, 2016 (RERA) is an Act of the Parliament of India which seeks to protect home-buyers as well as help boost investments in the real estate industry. The bill was passed by the Rajya Sabha on 10 March 2016 and by the Lok Sabha on 15 March 2016.",
    "What is stamp duty?": "Stamp duty is a tax that is levied on single property purchases or documents (including, historically, the majority of legal documents such as cheques, receipts, military commissions, marriage licences, land transactions etc.). A physical stamp had to be attached to or impressed upon the document to denote that stamp duty had been paid before the document was legally effective.",
    "What are the tax benefits on home loans?": "Under Section 80C of the Income Tax Act, you can claim deduction up to ₹1.5 lakh on principal repayment. Under Section 24(b), you can claim deduction up to ₹2 lakh on interest payment for self-occupied property. There is no limit on interest deduction for let-out properties.",
    "What is property registration?": "Property registration is the process of officially recording the transfer of property ownership with the government authorities. It involves paying stamp duty and registration charges, and executing a sale deed. Registration provides legal title to the buyer and is mandatory under the Registration Act, 1908.",
    "Can NRIs buy property in India?": "Yes, NRIs can purchase residential and commercial properties in India without any restrictions. However, they cannot purchase agricultural land, plantation property, or farmhouses without special permission from the Reserve Bank of India. NRIs can also inherit any immovable property in India."
}

for question, answer in FAQS.items():
    with st.expander(question):
        st.markdown(answer)
