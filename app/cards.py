import streamlit as st
import numpy as np
import pandas as pd

# ─────────────────────────────
# Utility
# ─────────────────────────────
def _ensure_chart_data():
    """Initialize demo data if not already present."""
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(
            np.random.randn(20, 3), columns=["LGBM", "XGBoost", "Tabular"]
        )


# ─────────────────────────────
# Model Overview Card
# ─────────────────────────────
def overview_card():
    _ensure_chart_data()

    with st.container(border=True):
        st.page_link("pages/model_overview.py", label="Model Overview", icon=":material/insights:")
        st.subheader("Ensemble Summary")

        cols = st.columns(3)
        cols[0].metric("Base Models", "3")
        cols[1].metric("Layers", "2")
        cols[2].metric("Accuracy", "94.7%")

        st.markdown("**Components**: LightGBM, XGBoost, Tabular NN")
        st.progress(0.7, text="Model Readiness")


# ─────────────────────────────
# Predictions Card
# ─────────────────────────────
def prediction_card():
    with st.container(border=True):
        # Add link to full dashboard
        st.page_link(
            "pages/visualization.py",
            label="Open Full Prediction Dashboard",
            icon=":material/open_in_new:"
        )

        st.caption("Snapshot of the latest prediction run.")



# ─────────────────────────────
# Data Visualization Card
# ─────────────────────────────
def data_card():
    _ensure_chart_data()

    with st.container(border=True):
        st.page_link("pages/visualization.py", label="Data Visualization", icon=":material/bar_chart:")
        st.subheader("Model Performance Summary")

        metrics = {
            "LightGBM": 0.0559,
            "XGBoost": 0.0558,
            "TabNet": 0.0612,
            "Stacked Ensemble": 0.08980,
        }
        df_metrics = pd.DataFrame(metrics.items(), columns=["Model", "RMSE"])
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        st.caption("Current validation scores across models.")



# ─────────────────────────────
# Data Analysis Card
# ─────────────────────────────
def data_analysis_card():
    with st.container(border=True):
        st.page_link("pages/data_analysis.py", label="Data Analysis", icon=":material/analytics:")
        st.subheader("Data Analysis")
        st.write("Explore the dataset used to train the models.")


# ─────────────────────────────
# Methodology Card
# ─────────────────────────────
def methodology_card():
    with st.container(border=True):
        st.page_link("pages/model_overview.py", label="Methodology", icon=":material/insights:")
        st.subheader("Architecture")

        st.markdown(
            """
            - **Layer 1**: Three independent models (LGBM, XGBoost, Tabular NN)
            - **Layer 2**: A meta-learner combines outputs for the final prediction
            """
        )
        st.caption("A stacked ensemble approach improves robustness.")


# ─────────────────────────────
# Input Features Card
# ─────────────────────────────
def input_features_card():
    with st.container(border=True):
        st.page_link("pages/model_overview.py", label="About", icon=":material/info:")
        st.subheader("Project Overview")

        st.write(
            """
            This dashboard demonstrates a machine learning model for **predicting road accident risk**
            using structured data.
            """
        )

        st.markdown(
            """
            - **Goal**: Predict accident probability from tabular data
            - **Stack**: Python, Scikit-learn, XGBoost
            - **Workflow**: Reproducible data prep and model training
            """
        )


# ─────────────────────────────
# Risk Interpretation Card
# ─────────────────────────────
def risk_interpretation_card():
    with st.container(border=True):
        st.page_link("pages/model_overview.py", label="Risk Interpretation", icon=":material/warning:")
        st.subheader("Interpretation")

        st.markdown(
            """
            - **Low**: 0.0 – 0.3  
            - **Medium**: 0.3 – 0.6  
            - **High**: 0.6 – 1.0
            """
        )
        st.caption("Normalized accident probability ranges.")
