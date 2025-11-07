import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
import numpy as np

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from road_accident_risk.modeling.base_models import load_models
from road_accident_risk.config import EXPECTED_COLUMNS

# ─────────────────────────────
# Page Configuration
# ─────────────────────────────
st.set_page_config(
    page_title="Prediction Visualization",
    page_icon=":material/bar_chart:",
    layout="wide",
)

# ─────────────────────────────
# Helper Functions
# ─────────────────────────────
def get_risk_level(prediction):
    """Categorize prediction into risk levels."""
    if prediction < 0.3:
        return "Low"
    elif prediction < 0.6:
        return "Medium"
    else:
        return "High"

def create_gauge_chart(prediction):
    """Create a gauge chart for the prediction."""
    level = get_risk_level(prediction)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {level}", 'font': {'size': 24}},
        delta={'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'green'},
                {'range': [0.3, 0.6], 'color': 'orange'},
                {'range': [0.6, 1], 'color': 'red'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction}
        }))
    return fig
def get_meta_importance():
    """Get feature importance (weights) from the meta-learner."""
    from road_accident_risk.modeling.meta_learner import load_meta_model

    try:
        meta_model = load_meta_model()
    except Exception as e:
        st.error(f"Failed to load meta-learner: {e}")
        return pd.DataFrame()

    # Determine feature importances
    if hasattr(meta_model, "coef_"):
        importances = meta_model.coef_.flatten()
    elif hasattr(meta_model, "feature_importances_"):
        importances = meta_model.feature_importances_
    else:
        st.warning("Meta model does not expose coefficients or feature_importances_.")
        return pd.DataFrame()

    # Manually assign proper base model names (replace these with your actual ones)
    base_model_names = ["XGB", "LGBM", "Tabular"]

    # Ensure the lengths match
    if len(base_model_names) != len(importances):
        base_model_names = [f"Model_{i}" for i in range(len(importances))]

    df = pd.DataFrame({
        "Base Model": base_model_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    return df


def create_meta_spider_chart(df_meta_importance):
    """Create a spider (radar) chart showing how the meta-learner weights base models."""
    if df_meta_importance.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df_meta_importance["Importance"],
        theta=df_meta_importance["Base Model"],
        fill="toself",
        name="Meta-Learner Weights",
        line_color="royalblue"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, df_meta_importance["Importance"].max()])
        ),
        showlegend=False,
        title="Meta-Learner Base Model Importance"
    )
    return fig

# ─────────────────────────────
# Main Application Flow
# ─────────────────────────────
def main():
    """Main function to run the Streamlit page."""
    st.title("Prediction Visualization Dashboard")

    # Get prediction from session state
    current_prediction = st.session_state.get('current_prediction', 0.0)

    # --- Risk Level Visualization ---
    st.header("Risk Level Visualization")
    with st.container(border=True):
        gauge_chart = create_gauge_chart(current_prediction)
        st.plotly_chart(gauge_chart, use_container_width=True)

    # --- Meta-Learner Weight Visualization ---
    st.header("Meta-Learner Weights")
    col1, col2 = st.columns([1, 1])  # 50-50 layout

    with col1:
        with st.container(border=True):
            df_meta_importance = get_meta_importance()
            if not df_meta_importance.empty:
                spider_chart = create_meta_spider_chart(df_meta_importance)
                st.plotly_chart(spider_chart, use_container_width=False)
            else:
                st.warning("Could not retrieve base model importances from meta-learner.")

    with col2:
        with st.container(border=True):
            st.markdown("### Interpretation")

            if not df_meta_importance.empty:
                top_model = df_meta_importance.iloc[0]['Base Model']
                top_weight = df_meta_importance.iloc[0]['Importance']
                total_weight = df_meta_importance['Importance'].sum()

                st.metric("Top Contributor", top_model, f"{top_weight:.2f}")

                # Relative share of total weights
                percent_share = (top_weight / total_weight) * 100 if total_weight > 0 else 0
                st.write(
                    f"The meta-learner relies most heavily on **{top_model}**, "
                    f"contributing roughly **{percent_share:.1f}%** of the total ensemble influence."
                )

                # Textual interpretation
                if percent_share > 60:
                    st.info(
                        f"The ensemble is largely driven by **{top_model}**, "
                        "suggesting it captures the dominant patterns in the data. "
                        "The meta-learner may be using other models primarily for fine-tuning."
                    )
                elif 30 <= percent_share <= 60:
                    st.info(
                        f"The meta-learner shows a balanced reliance on multiple base models, "
                        "indicating a well-regularized ensemble that benefits from model diversity."
                    )
                else:
                    st.info(
                        f"The ensemble evenly distributes importance across models, "
                        "highlighting complementary strengths among predictors."
                    )

                st.markdown("---")
                st.markdown("""
                **Interpretation Notes:**  
                - Larger weights imply greater trust in a base model’s predictions.  
                - A balanced distribution often means stronger generalization.  
                - Extreme dominance by one model may indicate redundancy or overfitting among others.
                """)
            else:
                st.warning("No importance data available for interpretation.")


if __name__ == "__main__":
    main()
