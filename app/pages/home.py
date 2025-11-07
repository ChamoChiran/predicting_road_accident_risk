import streamlit as st
from cards import methodology_card, input_features_card, risk_interpretation_card

# ─────────────────────────────
# Page configuration
# ─────────────────────────────
st.title(" Road Accident Risk Prediction")
st.caption("Data meets safety, uncovering how everyday conditions turn ordinary roads into danger zones.")

st.markdown("---")

# Welcome section
st.write("""
Welcome to the **Road Accident Risk Predictor**, an intelligent analytics platform 
that estimates the likelihood of road accidents based on infrastructure, 
environmental, and temporal factors.

Use the sidebar navigation to explore:
- **Model Overview** – Understand the ensemble architecture and data features.
- **Predictions** – Run the live model on your input data.
- **Data Visualization** – Explore insights from past records and trends.
""")

# ─────────────────────────────
# Overview Cards
# ─────────────────────────────
st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    methodology_card()

with col2:
    input_features_card()

with col3:
    risk_interpretation_card()

# ─────────────────────────────
# Footer
# ─────────────────────────────
st.markdown("---")
st.caption("© 2025 Road Accident Risk Prediction System — Built with Streamlit and Machine Learning")
