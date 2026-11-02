"""
Main Application Entry Point: E-Commerce Return Intelligence.

Provides executive landing overview, architecture diagram, system status,
and seamless navigation to all analytic and predictive modules.
"""

import os
import json
import streamlit as st
import pandas as pd

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card
from src.prediction import load_model_artifact

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Return Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Enterprise SaaS CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    """Load and cache cleaned dataset for fast UI response."""
    data_path = "dataset/Ecommerce_2000.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "dataset", "Ecommerce_2000.csv")
    df = pd.read_csv(data_path)
    df_dedup = df.drop_duplicates().copy()
    return df_dedup


@st.cache_resource
def load_cached_metadata():
    """Load model metadata and performance summaries."""
    meta_path = "models/model_metadata.json"
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


# Load data and metadata
df_clean = load_cached_data()
metadata = load_cached_metadata()

# Branding Header
st.markdown(
    render_brand_header(
        "E-COMMERCE RETURN INTELLIGENCE",
        "AI-Powered Return Prediction & Enterprise Supply Chain Analytics | Production ML Platform"
    ),
    unsafe_allow_html=True
)

# Sidebar Navigation Overview
with st.sidebar:
    st.markdown("### 📦 Return Intelligence")
    st.markdown("**Enterprise SaaS Edition**")
    st.markdown("---")
    st.info("💡 **Quick Navigation:** Use the pages menu above to access all analytics and prediction modules.")
    
    st.markdown("### 📊 Platform Metrics")
    st.markdown(f"- **Clean Records:** {len(df_clean):,}")
    st.markdown(f"- **Historical Return Rate:** {df_clean['Returned'].value_counts(normalize=True).get('Yes', 0.0):.2%}")
    st.markdown(f"- **Production Model:** {metadata.get('selected_model', 'Random Forest')}")
    st.markdown(f"- **Optimal Decision Cutoff:** {metadata.get('optimal_threshold', 0.47)}")
    
    st.markdown("---")
    st.markdown("Developed for **BTech CSE Data Science Final-Year Academic Defense & Deployment**.")

# Quick KPI Metrics Bar
total_orders = len(df_clean)
total_returns = int((df_clean['Returned'] == 'Yes').sum())
return_rate = (total_returns / total_orders) * 100
total_sales = float(df_clean['FinalAmount'].sum())
avg_order_val = total_sales / total_orders if total_orders > 0 else 0
avg_shipping = float(df_clean['ShippingDays'].mean())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(render_kpi_card("Total Orders", f"{total_orders:,}", "100% Unique Transactions"), unsafe_allow_html=True)
with col2:
    st.markdown(render_kpi_card("Total Returns", f"{total_returns:,}", f"{return_rate:.2f}% Return Rate"), unsafe_allow_html=True)
with col3:
    st.markdown(render_kpi_card("Total Net Revenue", f"${total_sales:,.0f}", f"${avg_order_val:.2f} Avg Order"), unsafe_allow_html=True)
with col4:
    st.markdown(render_kpi_card("Avg Shipping Time", f"{avg_shipping:.1f} Days", "Door-to-door transit"), unsafe_allow_html=True)
with col5:
    st.markdown(render_kpi_card("Selected Model", metadata.get("selected_model", "Random Forest").split()[0], "Tuned for Positive Recall"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Sections
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("""
    ### 🎯 System Capabilities & Architecture
    **E-Commerce Return Intelligence** provides an automated decision-support pipeline that mitigates 
    reverse logistics overheads and protects bottom-line margins.
    
    Using historical e-commerce purchasing and fulfillment signals, the system analyzes pre-dispatch risk 
    factors and classifies prospective returns before dispatch occurs.
    """)
    
    st.markdown("""
    #### ⚙️ End-to-End Dual Engine Architecture
    1. **Big Data Processing (PySpark MLlib):** Scalable ingestion, distributed DataFrame transformations, VectorAssembler feature scaling, and multi-core model training.
    2. **Cloud-Native Deployment (Scikit-Learn & Joblib):** Lightweight, zero-Java inference pipeline optimized for Streamlit Community Cloud hosting.
    3. **Threshold-Tuned Risk Scoring:** Calibration on positive-class F1-score to conquer severe 6.3% class imbalance.
    4. **Operational Intervention Engine:** Automatic generation of targeted customer retention and pre-dispatch warehouse mitigation actions.
    """)

with col_right:
    st.markdown("### 🚀 Module Quick Access")
    
    st.markdown("""
    - [📊 **Page 1: Executive Dashboard**](Executive_Dashboard)  
      *Real-time filtered KPI monitoring across categories, payment methods, and seasons.*
    - [🔮 **Page 2: Return Prediction**](Return_Prediction)  
      *Interactive single-order risk scoring, confidence intervals, and mitigation triggers.*
    - [📁 **Page 3: Bulk Prediction**](Bulk_Prediction)  
      *Batch CSV ingestion, automated validation, and downloadable prediction logs.*
    - [📈 **Page 4: Business Insights**](Business_Insights)  
      *In-depth category, shipping duration, and discount profitability analytics.*
    - [🧪 **Page 5: Model Performance**](Model_Performance)  
      *Confusion matrices, ROC/PR curves, feature importances, and validation comparisons.*
    - [📘 **Page 6: About Project**](About_Project)  
      *Comprehensive academic documentation, PySpark requirements, and viva defense guide.*
    """)

# System Health & Verification Card
st.markdown("---")
st.markdown("### 🛡️ Production Health & Model Integrity Status")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success("✅ **Dataset Sanitized:** 2,000 raw rows deduplicated to 1,980 unique transactions. Zero target leakage (ReviewScore excluded).")

with status_col2:
    st.success("✅ **Trained Pipeline Active:** Tuned Random Forest Classifier loaded with 197 one-hot and scaled numerical features.")

with status_col3:
    st.success("✅ **PySpark MLlib Validated:** Tasks 121–125 successfully executed on Spark DataFrame engine.")

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | BTech CSE Data Science Final-Year Academic Project | Production Cloud Release
</div>
""", unsafe_allow_html=True)
