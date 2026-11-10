\# PortIQ



\## AI-Powered Logistics Intelligence and Decision Support Platform



PortIQ is an interactive logistics operations platform designed to help operations teams inspect shipment data, identify operational risks, simulate disruptions and evaluate potential recovery actions.



\## Problem



Shipment disruptions can affect delivery deadlines, transport coordination and operational planning. Teams often need to manually inspect spreadsheets and combine information from different operational records.



PortIQ brings shipment intelligence, risk analysis, disruption simulation and AI-assisted explanations into one application.



\## Core Modules



1\. Shipment Intelligence Engine

2\. Disruption and Risk Intelligence

3\. BharatCode AI Operations Analyst

4\. What-if Disruption Simulator

5\. Recovery and Action Planner

6\. Decision Audit and Reporting

7\. Machine Learning Anomaly Detection



\## Technology



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Plotly

\- Scikit-learn

\- SQLite

\- BharatCode Qwen API



\## AI Integration



PortIQ uses the BharatCode Qwen model through its OpenAI-compatible API.



The application sends structured operational summaries to the model for explanations and decision support.



Operational calculations remain in the application. The AI layer explains the supplied evidence and does not independently modify shipment records.



\## Machine Learning



PortIQ uses Isolation Forest for unsupervised anomaly detection over shipment delay, cargo value, congestion, transport capacity and operational risk indicators.



An anomaly indicates an unusual data pattern. It is not proof of a real disruption.



\## Data



The application includes a reproducible simulated demonstration dataset and supports user-provided CSV and Excel files.



Demonstration records are not live shipment tracking data.



\## Run Locally



```bash

pip install -r requirements.txt

streamlit run app.py

### BharatCode AI Integration

PortIQ integrates the BharatCode Qwen 3.8 27B model through its OpenAI-compatible API.

The integration supports:
- AI-assisted logistics and shipment analysis.
- Operational disruption interpretation.
- Recovery strategy recommendations.
- Context-aware explanations of logistics risks.

The application includes API error handling and a local fallback to preserve core decision-support functionality when live inference is unavailable.

API credentials are configured through environment variables and are never included in the repository.