# AI Support Workflow Automation System

An agentic AI workflow automation system that simulates how businesses can use AI to handle customer support requests, reduce manual workload, and measure ROI.

## What It Does

This system processes a customer query through a multi-step AI workflow:

1. Intent classification
2. Knowledge base retrieval
3. Risk detection
4. Automation vs human escalation decision
5. ROI estimation
6. Query logging and analytics

## Why This Project Matters

Most AI demos stop at generating answers. This project goes further by asking:

- Should this query be automated?
- Is there enough confidence to respond?
- Is the request risky or sensitive?
- How much time and money could automation save?

## System Architecture
```text
Customer Query
    ↓
Intent Classifier
    ↓
Knowledge Base Retriever
    ↓
Risk Agent
    ↓
Decision Engine
    ↓
Response + ROI Metrics
    ↓
Logging + Analytics Dashboard
```

## Tech Stack
	•	Python
	•	scikit-learn
	•	pandas
	•	TF-IDF retrieval
	•	Streamlit
	•	Hugging Face Datasets
	•	CSV logging

## Key Features
	•	Intent classifier trained on customer-support data
	•	Knowledge-base retrieval from Markdown policy files
	•	Risk-based escalation logic
	•	ROI estimator for support automation
	•	Streamlit web demo
	•	Query logging for business analytics

## Example Output
```json
{
  "intent": "refund_request",
  "risk_level": "medium",
  "action": "auto_respond",
  "retrieval_score": 0.72,
  "estimated_minutes_saved": 6,
  "estimated_net_savings": 2.48
}
```

## Business Metrics Tracked
	•	Total queries
	•	Automation rate
	•	Escalation rate
	•	Average retrieval confidence
	•	Estimated minutes saved
	•	Estimated labor cost saved
	•	Estimated AI cost
	•	Estimated net savings


## How to Run
```bash
pip install -r requirements.txt
python -m src.load_dataset
python -m src.intent_classifier
streamlit run app/app.py
```

## AI Workflow Automation System for Customer Support

Python, scikit-learn, Streamlit, NLP, Data Analysis
	•	Designed and implemented an agentic AI system to automate customer support workflows, including intent classification, knowledge retrieval, and risk-based escalation
	•	Built a decision engine that determines whether queries should be auto-resolved or escalated, improving automation reliability
	•	Developed a retrieval system over structured policy documents using TF-IDF, enabling grounded, context-aware responses
	•	Implemented a risk detection module to flag sensitive queries (billing disputes, legal threats, medical concerns) for human review
	•	Created an ROI estimation framework to quantify automation impact, tracking time savings, labor cost reduction, and net savings per query
	•	Built an interactive Streamlit dashboard to simulate real-time support workflows and visualize system outputs
	•	Logged and analyzed query data to compute automation rate, escalation rate, and total cost savings