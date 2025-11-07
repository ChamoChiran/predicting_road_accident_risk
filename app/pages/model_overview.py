import streamlit as st

st.title("Model Overview")

st.write("""
The Road Accident Risk Predictor is powered by a **stacked ensemble architecture**
a layered approach that combines the strengths of multiple machine learning models
to produce a unified, reliable risk estimate.
""")

st.divider()

st.subheader("Base Models — Layer 1")
st.write("""
The first layer consists of three complementary algorithms, each designed to capture
different structures and interactions within the data:
""")

st.markdown("""
- **LightGBM** — A gradient boosting framework optimized for speed and efficiency.  
  Excels at handling large, tabular datasets with minimal preprocessing.
- **XGBoost** — An enhanced boosting algorithm known for its precision and 
  fine-grained control over model complexity.
- **Tabular Neural Network** — A feedforward neural model that learns nonlinear 
  relationships and smooth interactions among continuous features.
""")

st.divider()

st.subheader("Meta-Learner — Layer 2")
st.write("""
The second layer, called the **meta-learner**, receives the predictions from the base models.
It learns how to weight and combine those predictions, producing a balanced final output that
reduces individual model biases and improves overall generalization.
""")

st.divider()

st.subheader("Why Stacking?")
st.write("""
Ensemble stacking offers a pragmatic balance between interpretability and performance.  
By leveraging the diversity of multiple algorithms, it achieves:
""")

st.markdown("""
- **Higher accuracy** through model complementarity  
- **Better stability** across varying data distributions  
- **Reduced overfitting** compared to relying on a single model  
""")

st.caption("This architecture reflects a data-centric design philosophy — robust, interpretable, and adaptive.")
