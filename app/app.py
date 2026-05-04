import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.workflow import SupportWorkflow
from src.logger import log_result


st.set_page_config(page_title="AI Support Workflow", layout="wide")

st.title("AI Support Workflow Automation System")

st.markdown("""
This system simulates an AI-powered customer support workflow:
- Intent classification
- Knowledge retrieval
- Risk detection
- Automation vs escalation decision
- ROI estimation
""")

workflow = SupportWorkflow(domain="ecommerce")

query = st.text_input("Enter a customer query:")

if st.button("Run Workflow") and query:
    result = workflow.run(query)

    log_result(result)

    st.subheader("Workflow Output")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Intent", result["intent"])
        st.metric("Intent Confidence", result["intent_confidence"])
        st.metric("Risk Level", result["risk_level"])
        st.metric("Action", result["action"])

    with col2:
        st.metric("Retrieval Score", result["retrieval_score"])
        st.metric("Source", result["retrieved_source"])

    st.subheader("ROI Metrics")
    st.json(result["roi"])

    st.subheader("Response")
    st.write(result["response"])

    st.subheader("Top Retrieved Chunks")
    for chunk in result["retrieved_chunks"]:
        st.markdown(f"**{chunk['source_file']} (score: {chunk['score']})**")
        st.write(chunk["text"])