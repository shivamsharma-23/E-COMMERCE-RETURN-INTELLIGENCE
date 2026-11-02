"""
Page 3: Bulk Order Prediction for E-Commerce Return Intelligence.

Batch CSV ingestion, automated schema validation, missing-value imputation,
interactive risk classification tables, summary metrics, and prediction exports.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card
from src.prediction import load_model_artifact, predict_bulk_orders

st.set_page_config(page_title="Bulk Order Prediction | Return Intelligence", page_icon="📁", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.joblib")
    return load_model_artifact(model_path)


model_artifact = get_model()

st.markdown(
    render_brand_header(
        "BATCH & BULK ORDER RETURN PREDICTION",
        "Upload Order Manifests (CSV), Score Fleet Return Probabilities, and Export Mitigation Reports"
    ),
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Batch Settings")
    default_thresh = float(model_artifact.get('threshold', 0.47))
    custom_threshold = st.slider(
        "Batch Decision Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=default_thresh,
        step=0.01,
        help="Probability cutoff separating RETURN from NOT RETURN."
    )
    st.markdown("---")
    st.markdown("### 📥 Download Sample Template")
    
    sample_path = "output/sample_bulk_orders.csv"
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "sample_bulk_orders.csv")
    
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                label="📄 Download 30-Order Sample CSV",
                data=f.read(),
                file_name="sample_bulk_orders.csv",
                mime="text/csv",
                use_container_width=True
            )
    st.caption("Use this template to test bulk predictions without formatting headaches.")

# Main Ingestion Container
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Order Batch File (CSV)")
st.write("Upload your e-commerce batch file. The upload does **not** require the `Returned` column.")

uploaded_file = st.file_uploader(
    "Drag and drop your CSV file here",
    type=["csv"],
    help="Accepts CSV with order columns: ProductCategory, UnitPrice, Quantity, DiscountPercent, PaymentMethod, etc."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to parse CSV file: {e}")
        st.stop()

    st.success(f"✅ Successfully loaded `{uploaded_file.name}` containing **{len(df_uploaded):,}** rows and **{len(df_uploaded.columns)}** columns.")

    # Column Validation Check
    critical_cols = ['ProductCategory', 'UnitPrice', 'Quantity']
    missing_critical = [c for c in critical_cols if c not in df_uploaded.columns]

    if missing_critical:
        st.error(f"⚠️ Missing required core columns: {missing_critical}. Please ensure your CSV includes these fields.")
        st.stop()

    # Run Prediction
    with st.spinner("⚡ Running AI Return Prediction Pipeline..."):
        results_df, summary = predict_bulk_orders(
            df_uploaded,
            model_artifact=model_artifact,
            custom_threshold=custom_threshold
        )

    # Summary KPI Cards
    st.markdown("### 📈 Batch Risk Summary")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(render_kpi_card("Total Uploaded", f"{summary['total_orders']:,}", "Manifest records"), unsafe_allow_html=True)
    with kpi2:
        st.markdown(render_kpi_card("Predicted Returns", f"{summary['predicted_returns']:,}", f"{summary['predicted_return_rate']}% Return Rate"), unsafe_allow_html=True)
    with kpi3:
        st.markdown(render_kpi_card("High Risk Orders", f"{summary['high_risk_count']:,}", "Prob ≥ 50%"), unsafe_allow_html=True)
    with kpi4:
        st.markdown(render_kpi_card("Medium Risk", f"{summary['medium_risk_count']:,}", "25% - 50% Prob"), unsafe_allow_html=True)
    with kpi5:
        st.markdown(render_kpi_card("At-Risk Revenue", f"${summary['potential_returned_revenue']:,.2f}", "Exposure volume"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk Distribution Chart & Category Breakdown
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Risk Category Distribution")
        risk_counts = results_df['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Category', 'Count']
        
        color_map = {
            'Low Risk': '#16A34A',
            'Medium Risk': '#D97706',
            'High Risk': '#DC2626'
        }
        fig_risk = px.pie(
            risk_counts,
            names='Risk_Category',
            values='Count',
            color='Risk_Category',
            color_discrete_map=color_map,
            hole=0.55
        )
        fig_risk.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 📦 Predicted Returns by Category")
        cat_ret = results_df.groupby('ProductCategory')['Predicted_Return'].apply(
            lambda x: (x == 'RETURN').sum()
        ).reset_index(name='Predicted_Returns').sort_values('Predicted_Returns', ascending=False)

        fig_cat = px.bar(
            cat_ret,
            x='ProductCategory',
            y='Predicted_Returns',
            color='Predicted_Returns',
            color_continuous_scale=[[0, '#93C5FD'], [1, '#1E3A8A']],
            text='Predicted_Returns'
        )
        fig_cat.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Interactive Prediction Table
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Prediction Results Data Table")
    
    # Filter table view by risk
    filter_choice = st.radio(
        "Filter Display Table:",
        options=["All Orders", "High Risk Only", "Returns Only"],
        horizontal=True
    )
    
    display_df = results_df.copy()
    if filter_choice == "High Risk Only":
        display_df = display_df[display_df['Risk_Category'] == 'High Risk']
    elif filter_choice == "Returns Only":
        display_df = display_df[display_df['Predicted_Return'] == 'RETURN']

    display_cols = [
        c for c in [
            'OrderID', 'CustomerID', 'ProductCategory', 'ProductSubCategory',
            'UnitPrice', 'Quantity', 'FinalAmount', 'Return_Probability_Pct',
            'Predicted_Return', 'Risk_Category', 'Recommended_Action'
        ] if c in display_df.columns
    ]

    st.dataframe(display_df[display_cols], use_container_width=True, height=350)

    # Download Output CSV
    csv_bytes = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Predictions CSV Report",
        data=csv_bytes,
        file_name=f"return_predictions_{uploaded_file.name}",
        mime="text/csv",
        type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("💡 **Ready for input:** Please upload a CSV file above or download the sample 30-order template to test batch prediction.")

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Batch Processing Pipeline
</div>
""", unsafe_allow_html=True)
