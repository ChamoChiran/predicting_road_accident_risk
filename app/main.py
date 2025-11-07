# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
from cards import overview_card, prediction_card, data_card, data_analysis_card

st.set_page_config(layout="wide")

st.set_page_config(
    page_title="Road Accident Risk",
    page_icon="🚦",
    layout="wide"
)


# Custom CSS to adjust sidebar width
# initialize demo data for sidebar previews
if "init" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(
        np.random.randn(20, 3), columns=["LGBM", "XGB", "Tabular"]
    )
    st.session_state.init = True

# define your pages
pages = [
    st.Page("pages/home.py", title="Home", icon=":material/home:"),
    st.Page("pages/data_analysis.py", title="Data Analysis", icon=":material/analytics:"),
    st.Page("pages/model_overview.py", title="Model Overview", icon=":material/insights:"),
    st.Page("pages/predictions.py", title="Predictions", icon=":material/analytics:"),
    st.Page("pages/visualization.py", title="Data Visualization", icon=":material/bar_chart:"),
]

page = st.navigation(pages)
page.run()

# sidebar preview area
with st.sidebar.container(height=400):
    if page.title == "Model Overview":
        overview_card()
    elif page.title == "Predictions":
        prediction_card()
    elif page.title == "Data Analysis":
        data_analysis_card()
    elif page.title == "Data Visualization":
        data_card()
    else:
        st.page_link("pages/home.py", label="Home", icon=":material/home:")
        st.write("Welcome! Select a page to explore the accident risk predictor.")
