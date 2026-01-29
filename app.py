# paste the README text here
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="💰"
)

# Custom styling
st.markdown("""
    <style>
    body {background-color:#0e1117; color:#fff;}
    .sidebar .sidebar-content {background-color:#1c1c24;}
    h1, h2, h3, h4 { color: #F9FAFB !important; }
    .stNumberInput label, .stSelectbox label {
        color: #E0E0E0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Predict YouTube Ad Revenue")

# Load model
MODEL_PATH = 'I:/Project/linear_regression_model.pkl'
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check the path.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Sidebar for file upload
st.sidebar.header("/content/Content-Monetization-Modeler/youtube_ad_revenue_dataset.csv")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Manual input section
st.subheader("✍️ Enter Video Feature Values")

feature_cols = [
    'video_duration_minutes',
    'views',
    'likes',
    'comments',
    'subscribers_gained',
    'avg_view_duration_seconds'
]

input_data = {}
cols_per_row = 3
rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(feature_cols):
            col_name = feature_cols[idx]
            input_data[col_name] = cols[j].number_input(f"{col_name}", value=0.0)

# Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)
        st.success(f"💰 **Predicted Ad Revenue:** ${prediction:,.2f}")
        if prediction == 0:
            st.info("This video may not generate ad revenue based on the current input values.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Optional: show uploaded file preview
if uploaded_file is not None:
    st.subheader("📄 Uploaded File Preview")
    df_uploaded = pd.read_csv(uploaded_file)
    st.dataframe(df_uploaded.head())
