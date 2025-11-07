import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_accident_risk.modeling.base_models import predict_from_user_input
from road_accident_risk.modeling.meta_learner import predict_from_stage1_output

# ─────────────────────────────
# Page Configuration
# ─────────────────────────────
st.set_page_config(
    page_title="Predictions",
    page_icon=":material/analytics:",
    layout="wide",
)

# ─────────────────────────────
# UI Components
# ─────────────────────────────
def get_user_input():
    """Display input fields and return a dictionary of user selections."""
    st.subheader("Enter Scenario Details")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            road_type = st.selectbox("Road Type", ["highway", "urban", "rural"])
            num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=8, value=2)
            curvature = st.number_input("Curvature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            speed_limit = st.number_input("Speed Limit", min_value=20, max_value=120, value=60, step=5)

        with col2:
            lighting = st.selectbox("Lighting", ["daylight", "dusk", "night"])
            weather = st.selectbox("Weather", ["clear", "rainy", "snowy", "foggy"])
            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
            num_reported_accidents = st.number_input("Num Reported Accidents (Area)", min_value=0, value=0)

        with col3:
            road_signs_present = st.checkbox("Road Signs Present", value=False)
            public_road = st.checkbox("Public Road", value=False)
            holiday = st.checkbox("Holiday", value=True)
            school_season = st.checkbox("School Season", value=True)

    user_input = {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "lighting": lighting,
        "weather": weather,
        "road_signs_present": road_signs_present,
        "public_road": public_road,
        "time_of_day": time_of_day,
        "holiday": holiday,
        "school_season": school_season,
        "num_reported_accidents": num_reported_accidents,
    }
    return user_input

def display_predictions(predictions: pd.DataFrame, final_prediction: float):
    """Display the base and final predictions."""
    st.subheader("Prediction Results")

    with st.container(border=True):
        st.metric("Final Predicted Risk", f"{final_prediction:.2%}", "inverse")

        with st.expander("Show Base Model Predictions"):
            display_df = predictions.copy()
            display_df["stacked_prediction"] = final_prediction
            st.dataframe(display_df, use_container_width=True)

# ─────────────────────────────
# Main Application Flow
# ─────────────────────────────
def main():
    """Main function to run the Streamlit page."""
    st.title("Accident Risk Prediction")

    user_input = get_user_input()

    if st.button("Predict Risk", type="primary"):
        # Stage 1: Get base model predictions
        stage1_output = predict_from_user_input(user_input)

        # Stage 2: Get final prediction from meta-learner
        final_prediction = predict_from_stage1_output(stage1_output)

        # Store in session state for other pages
        st.session_state["current_prediction"] = float(final_prediction[0])
        st.session_state["stage1_output"] = stage1_output

        display_predictions(stage1_output, final_prediction[0])

if __name__ == "__main__":
    main()
