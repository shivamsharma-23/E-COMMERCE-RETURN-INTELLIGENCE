"""
Page 5: Model Performance & Academic Evaluation for E-Commerce Return Intelligence.

Displays validation vs test metrics, side-by-side model comparison,
confusion matrices, ROC and PR curves, feature importance rankings,
and PySpark MLlib distributed performance benchmarks.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card
from src.visualizations import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance
)

st.set_page_config(page_title="Model Performance | Return Intelligence", page_icon="🧪", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_performance_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def get_json(rel_path):
        p = os.path.join(base_dir, rel_path)
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return {}

    eval_data = get_json("models/evaluation_results.json")
    meta_data = get_json("models/model_metadata.json")
    feat_data = get_json("models/feature_importance.json")
    spark_data = get_json("models/spark_metrics.json")
    return eval_data, meta_data, feat_data, spark_data


eval_results, metadata, feature_imp, spark_metrics = load_performance_artifacts()

st.markdown(
    render_brand_header(
        "MACHINE LEARNING MODEL PERFORMANCE & EVALUATION",
        "Rigorous Validation Benchmarks, Class Imbalance Handling & Untouched Test Set Evaluation"
    ),
    unsafe_allow_html=True
)

# Top Bar: Model Selection & Split Details
sel_model = eval_results.get("selected_model", "Random Forest Classifier")
threshold = eval_results.get("threshold", 0.47)
reason = metadata.get("selection_reason", "Selected based on superior validation F1-score and PR-AUC.")

st.markdown('<div class="content-card">', unsafe_allow_html=True)
col_top1, col_top2 = st.columns([2, 1])

with col_top1:
    st.markdown(f"### 🏆 Selected Production Model: **{sel_model}**")
    st.markdown(f"**Academic Selection Rational:** {reason}")
    st.markdown(f"**Tuned Decision Threshold ($\tau$):** `{threshold}` *(Tuned specifically on Validation Split to maximize minority-class F1-score)*")

with col_top2:
    st.markdown("#### 📐 Stratified Split Partitions")
    st.markdown("- **Train Set (70%):** 1,386 orders (88 returns, 6.35%)")
    st.markdown("- **Validation Set (15%):** 297 orders (19 returns, 6.40%)")
    st.markdown("- **Untouched Test Set (15%):** 297 orders (19 returns, 6.40%)")
st.markdown('</div>', unsafe_allow_html=True)

# Toggle View: Validation vs Untouched Test Set
st.markdown("### 📊 Performance Metric Assessment")
metric_stage = st.radio(
    "Select Evaluation Stage to Inspect:",
    options=["Untouched Final Test Set (Generalization Test)", "Validation Set (Hyperparameter & Threshold Tuning)"],
    horizontal=True
)

is_test = "Test" in metric_stage
metrics_key = "test_metrics" if is_test else "validation_metrics"
stage_label = "Final Test" if is_test else "Validation"

lr_m = eval_results.get(metrics_key, {}).get("Logistic Regression", {})
rf_m = eval_results.get(metrics_key, {}).get("Random Forest", {})
sel_m = rf_m if sel_model == "Random Forest Classifier" else lr_m

# KPI Metrics for Selected Model
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
with kpi1:
    st.markdown(render_kpi_card(f"{stage_label} Accuracy", f"{sel_m.get('accuracy', 0)*100:.1f}%", "Overall accuracy"), unsafe_allow_html=True)
with kpi2:
    st.markdown(render_kpi_card(f"{stage_label} Precision", f"{sel_m.get('precision', 0)*100:.1f}%", "Positive class (Returned)"), unsafe_allow_html=True)
with kpi3:
    st.markdown(render_kpi_card(f"{stage_label} Recall", f"{sel_m.get('recall', 0)*100:.1f}%", "Caught return share"), unsafe_allow_html=True)
with kpi4:
    st.markdown(render_kpi_card(f"{stage_label} F1-Score", f"{sel_m.get('f1_score', 0):.4f}", "Harmonic precision-recall"), unsafe_allow_html=True)
with kpi5:
    st.markdown(render_kpi_card(f"{stage_label} ROC-AUC", f"{sel_m.get('roc_auc', 0):.4f}", "Discrimination power"), unsafe_allow_html=True)
with kpi6:
    st.markdown(render_kpi_card(f"{stage_label} PR-AUC", f"{sel_m.get('pr_auc', 0):.4f}", "Minority-class curve"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Side-by-Side Model Comparison Table
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown(f"### ⚖️ Side-by-Side Model Comparison ({stage_label} Stage)")

comp_df = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Logistic Regression (Tuned)": f"{lr_m.get('accuracy', 0)*100:.2f}%",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('accuracy', 0)*100:.2f}%",
        "Optimal Direction": "Higher"
    },
    {
        "Metric": "Precision (Class 1: Returned)",
        "Logistic Regression (Tuned)": f"{lr_m.get('precision', 0)*100:.2f}%",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('precision', 0)*100:.2f}%",
        "Optimal Direction": "Higher"
    },
    {
        "Metric": "Recall (Class 1: Returned)",
        "Logistic Regression (Tuned)": f"{lr_m.get('recall', 0)*100:.2f}%",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('recall', 0)*100:.2f}%",
        "Optimal Direction": "Higher"
    },
    {
        "Metric": "F1-Score (Class 1)",
        "Logistic Regression (Tuned)": f"{lr_m.get('f1_score', 0):.4f}",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('f1_score', 0):.4f}",
        "Optimal Direction": "Higher"
    },
    {
        "Metric": "ROC-AUC",
        "Logistic Regression (Tuned)": f"{lr_m.get('roc_auc', 0):.4f}",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('roc_auc', 0):.4f}",
        "Optimal Direction": "Higher"
    },
    {
        "Metric": "PR-AUC (Precision-Recall Area)",
        "Logistic Regression (Tuned)": f"{lr_m.get('pr_auc', 0):.4f}",
        "Random Forest Classifier (Tuned)": f"{rf_m.get('pr_auc', 0):.4f}",
        "Optimal Direction": "Higher"
    }
])
st.table(comp_df)
st.caption(
    "⚠️ **Academic Note on Class Imbalance (6.3% returns):** Standard unweighted accuracy is misleading because a dummy model "
    "predicting 'Not Returned' for every record achieves 93.6% accuracy while identifying 0 returns. We explicitly prioritize "
    "PR-AUC, F1-score, and positive-class Recall."
)
st.markdown('</div>', unsafe_allow_html=True)

# Confusion Matrices
st.markdown("### 🧩 Confusion Matrices (Untouched Test Set)")
cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_confusion_matrix(
            rf_m.get("confusion_matrix", {}),
            title="Random Forest (Selected Model) - Test Set"
        ),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with cm_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_confusion_matrix(
            lr_m.get("confusion_matrix", {}),
            title="Logistic Regression (Baseline) - Test Set"
        ),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ROC and PR Curves
st.markdown("### 📈 Discrimination Curves (Untouched Test Set)")
curves_data = eval_results.get("curves", {})
curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    lr_roc_auc = float(eval_results.get("test_metrics", {}).get("Logistic Regression", {}).get("roc_auc", 0.5))
    rf_roc_auc = float(eval_results.get("test_metrics", {}).get("Random Forest", {}).get("roc_auc", 0.5))
    st.plotly_chart(plot_roc_curves(curves_data, lr_roc_auc, rf_roc_auc), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with curve_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_pr_curves(curves_data, baseline_rate=0.064), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature Importance
st.markdown("### 🔍 Model Interpretability & Feature Importance")
feat_col1, feat_col2 = st.columns([1, 1])

with feat_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    top_detailed = feature_imp.get("detailed", [])
    st.plotly_chart(plot_feature_importance(top_detailed, top_n=12), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with feat_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 💡 Aggregated High-Level Predictive Drivers")
    agg_list = feature_imp.get("aggregated", [])
    if agg_list:
        agg_df = pd.DataFrame(agg_list).rename(columns={'feature': 'Feature Domain', 'importance': 'Importance Share'})
        st.dataframe(agg_df, use_container_width=True, height=280)
    st.info("""
    **Core Predictive Insights:**
    - **UnitPrice & EffectiveUnitPrice (~15% importance):** Higher priced purchases carry significantly elevated return friction.
    - **TotalAmount & FinalAmount (~12% importance):** Basket expenditure volume influences customer retention sensitivity.
    - **ShippingPerDollar & ShippingDays (~8% importance):** Disproportionate delivery delays relative to order spend directly provoke buyer remorse.
    - **ProductCategory_Clothing:** By far the single most predictive categorical feature on return probability.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# PySpark Big Data MLlib Benchmark Section
if spark_metrics:
    st.markdown("---")
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ PySpark Big Data MLlib Distributed Pipeline Benchmarks")
    st.markdown(f"**Execution Framework:** `{spark_metrics.get('framework')}` | **Mode:** `{spark_metrics.get('execution_mode')}`")

    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        st.markdown("#### 🌲 Spark MLlib Random Forest")
        sp_rf = spark_metrics.get("models", {}).get("Random Forest", {})
        st.json(sp_rf)

    with sp_col2:
        st.markdown("#### 📈 Spark MLlib Logistic Regression")
        sp_lr = spark_metrics.get("models", {}).get("Logistic Regression", {})
        st.json(sp_lr)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Scientific Evaluation Suite
</div>
""", unsafe_allow_html=True)
