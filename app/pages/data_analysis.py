import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ─────────────────────────────
# Streamlit Page Config
# ─────────────────────────────
st.set_page_config(page_title="EDA | Road Accident Risk", layout="wide")

st.title("Exploratory Data Analysis – Road Accident Risk Dataset")
st.markdown("""
This dashboard explores how **lighting, weather, road type, speed limit, and curvature**
influence the risk of road accidents.  
All plots are interactive and use Streamlit’s native charting style.
""")

# ─────────────────────────────
# Load Dataset
# ─────────────────────────────
@st.cache_data
def load_data(sample_size=10000):
    path = "../app/data/df.csv"
    try:
        # Read the header to get column names
        header = pd.read_csv(path, nrows=0).columns
        # Count total number of rows
        n = sum(1 for _ in open(path)) - 1
        # If sample size is larger than total rows, read all data
        if sample_size >= n:
            return pd.read_csv(path)
        # Generate rows to skip
        skip = sorted(random.sample(range(1, n + 1), n - sample_size))
        # Read the data, skipping the specified rows
        return pd.read_csv(path, skiprows=skip, header=0, names=header)
    except FileNotFoundError:
        return None

data = load_data()
if data is None:
    st.error("Dataset not found. Please ensure `data/processed/train.csv` exists.")
    st.stop()

st.success(f"Loaded a random sample of {data.shape[0]:,} rows and {data.shape[1]:,} columns.")

# ─────────────────────────────
# Display Data Table
# ─────────────────────────────
st.subheader("Raw Data")
st.dataframe(data.head(100))

# ─────────────────────────────
# Distribution of Accident Risk
# ─────────────────────────────
st.subheader("How is accident risk distributed?")
st.markdown("Log-transformation helps us see small variations more clearly.")

tab1, tab2 = st.tabs(["Original", "Log-transformed"])

with tab1:
    fig, ax = plt.subplots()
    ax.hist(data["accident_risk"], bins=50, color="#7E7EFF", edgecolor="black")
    ax.set_title("Distribution of Accident Risk")
    ax.set_xlabel("Accident Risk")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    ax.hist(np.log1p(data["accident_risk"]), bins=50, color="#7EFFB2", edgecolor="black")
    ax.set_title("Distribution of Log-Transformed Accident Risk")
    ax.set_xlabel("Log(1 + Accident Risk)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

st.markdown("""
> Most records show low accident risk, but a long tail of high-risk cases exists.  
> After log transformation, the data becomes more balanced and reveals mid-range variation.
""")

# ─────────────────────────────
# Lighting Conditions
# ─────────────────────────────
if "lighting" in data.columns:
    st.subheader("Does lighting affect accident risk?")
    lighting_summary = (
        data.groupby("lighting")["accident_risk"]
        .mean()
        .reset_index()
        .sort_values("accident_risk", ascending=False)
    )

    st.bar_chart(
        data=lighting_summary,
        x="lighting",
        y="accident_risk",
        use_container_width=True,
    )

    st.markdown("""
    > Poor lighting (dusk, night, fog) generally leads to higher accident risk,
    reinforcing the importance of visibility and reflective road markings.
    """)

# ─────────────────────────────
# Weather Conditions
# ─────────────────────────────
if "weather" in data.columns:
    st.subheader("How does weather influence accident risk?")
    weather_summary = (
        data.groupby("weather")["accident_risk"]
        .mean()
        .reset_index()
        .sort_values("accident_risk", ascending=False)
    )

    st.bar_chart(
        data=weather_summary,
        x="weather",
        y="accident_risk",
        use_container_width=True,
    )

    st.markdown("""
    > Rain, fog, and storm conditions sharply increase risk compared to clear weather.  
    > This pattern often matches reduced visibility and slippery road surfaces.
    """)

# ─────────────────────────────
# Road Type
# ─────────────────────────────
if "road_type" in data.columns:
    st.subheader("Do different road types change risk?")
    road_summary = (
        data.groupby("road_type")["accident_risk"]
        .mean()
        .reset_index()
        .sort_values("accident_risk", ascending=False)
    )

    st.bar_chart(
        data=road_summary,
        x="road_type",
        y="accident_risk",
        use_container_width=True,
    )

    st.markdown("""
    > Complex intersections or rural roads tend to show higher risk values,  
    while well-lit highways remain relatively safer on average.
    """)

# ─────────────────────────────
# Speed Limit
# ─────────────────────────────
if "speed_limit" in data.columns:
    st.subheader("How does speed limit relate to accident risk?")
    speed_summary = (
        data.groupby("speed_limit")["accident_risk"]
        .mean()
        .reset_index()
        .sort_values("speed_limit")
    )

    st.line_chart(
        data=speed_summary,
        x="speed_limit",
        y="accident_risk",
        use_container_width=True,
    )

    st.markdown("""
    > As speed limits rise, average accident risk generally increases —  
    but the relationship may not be perfectly linear, hinting at driver adaptation effects.
    """)

# ─────────────────────────────
# Road Curvature
# ─────────────────────────────
if "curvature" in data.columns:
    st.subheader("Does road curvature affect risk?")
    curvature_df = data[["curvature", "accident_risk"]].copy()

    st.scatter_chart(
        curvature_df,
        x="curvature",
        y="accident_risk",
        use_container_width=True,
    )

    st.markdown("""
    > Mild curvature has little effect, but sharper bends correlate with spikes in risk.  
    > This likely reflects traction loss and reaction time issues on curves.
    """)

# ─────────────────────────────
# Footer
# ─────────────────────────────
st.markdown("---")
st.caption("© 2025 Road Accident Risk – Chamodh - EDA Dashboard")
