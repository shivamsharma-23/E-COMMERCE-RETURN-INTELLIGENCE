"""
Page 6: About Project & Academic Viva Defense Guide.

Comprehensive technical documentation tailored for final-year BTech CSE
Data Science academic evaluations, technical project vivas, and architectural defenses.
"""

import os
import streamlit as st

from src.styles import CUSTOM_CSS, render_brand_header

st.set_page_config(page_title="About Project & Academic Defense | Return Intelligence", page_icon="📘", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    render_brand_header(
        "ABOUT PROJECT & ACADEMIC VIVA DEFENSE GUIDE",
        "Complete Technical Specification, Mathematical Formulations, Architecture & Oral Exam Cheat-Sheet"
    ),
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Project Specification & Problem Statement",
    "⚙️ Pipeline Architecture & Methodology",
    "⚡ Big Data PySpark vs Scikit-Learn Engine",
    "🎓 Academic Viva Q&A Defense Guide"
])

with tab1:
    st.markdown("""
    ### 1. Problem Statement
    In modern e-commerce, product returns represent an existential margin erosion threat. Retailers report 
    global return rates averaging 15% to 30%, incurring colossal reverse logistics fees, warehouse inspection 
    burdens, inventory obsolescence, and packaging waste.
    
    Traditional e-commerce platforms handle returns **post-facto** — reacting only after an RMA (Return Merchandise 
    Authorization) request is logged. 
    
    ### 2. Project Objective
    **E-Commerce Return Intelligence** shifts returns management from a reactive cost center to a **proactive, 
    pre-dispatch predictive risk intelligence platform**. By predicting return probability at the point of order creation, 
    enterprises can trigger targeted mitigation strategies (e.g., fit verification, premium packaging, delivery milestones) 
    before the parcel departs the warehouse.

    ### 3. Empirical Dataset Profile
    - **Source File:** `Ecommerce_2000.csv`
    - **Raw Observations:** 2,000 records across 23 columns
    - **Clean Observations:** 1,980 unique records (20 exact duplicates eliminated)
    - **Target Variable:** `Returned` ('Yes' = 1, 'No' = 0)
    - **Target Distribution:** 126 Returns (6.36%) vs 1,854 Non-Returns (93.64%)
    - **Class Imbalance Ratio:** ~14.7 : 1 (Extreme minority positive class)
    - **Data Leakage Prevention:** `ReviewScore` was deliberately stripped prior to modeling because customer reviews are recorded days or weeks *after* order receipt and often correlate with dissatisfaction, which would cause severe target leakage. Identifiers (`OrderID`, `CustomerID`, `FirstName`, `LastName`) were likewise discarded to ensure algorithmic generalizability.
    """)

with tab2:
    st.markdown("""
    ### 🛠️ End-to-End Pipeline Methodology

    #### A. Data Preprocessing & Defensive Imputation
    - **Deduplication:** Dropped exact duplicated transaction records.
    - **Defensive Imputation:** Numerical features imputed with their median (robust against extreme outliers); categorical features imputed with 'Missing' / mode.
    - **Target Binarization:** Mapped string categorical labels to numeric binary integers.

    #### B. Domain-Driven Feature Engineering
    From raw transactional fields, 7 domain-specific behavioral and financial ratios were formulated:
    1. **Discount Ratio:** $\\text{DiscountRatio} = \\frac{\\text{DiscountAmount}}{\\text{TotalAmount} + \\epsilon}$
    2. **Effective Unit Price:** $\\text{EffectiveUnitPrice} = \\frac{\\text{FinalAmount}}{\\text{Quantity} + \\epsilon}$
    3. **Shipping per Dollar:** $\\text{ShippingPerDollar} = \\frac{\\text{ShippingDays}}{\\text{FinalAmount} + \\epsilon}$
    4. **Aggressive Discount Indicator:** $\\mathbb{I}(\\text{DiscountPercent} \\ge 20)$
    5. **Bulk Order Indicator:** $\\mathbb{I}(\\text{Quantity} \\ge 3)$
    6. **Order Month:** Extracted temporal seasonality signal from `OrderDate`.
    7. **Weekend Shopping Flag:** Derived indicator for Saturday/Sunday checkout behavior.

    #### C. Stratified Splitting Architecture
    To preserve the fragile 6.36% positive class ratio across all phases:
    - **Training Set (70% - 1,386 orders):** Used exclusively for model weight learning and feature transformation fits.
    - **Validation Set (15% - 297 orders):** Reserved strictly for hyperparameter tuning, model comparison, and decision threshold calibration ($\tau$).
    - **Untouched Test Set (15% - 297 orders):** Kept completely isolated until final publication to verify real-world generalization performance.

    #### D. Handling Severe Class Imbalance
    Standard estimators optimize for zero-one loss, producing dummy models with 93.6% accuracy but 0.0% return recall. We solved this with:
    1. **Balanced Class Weighting:** Penalizing positive class false negatives inversely proportional to class frequencies:
       $$w_1 = \\frac{N}{2 \\times N_1}$$
    2. **Empirical Threshold Optimization:** Rather than the naive 0.50 cutoff, we optimized the decision cutoff $\\tau \\in [0.10, 0.90]$ on the validation set to maximize minority-class $F_1$-score.
    """)

with tab3:
    st.markdown("""
    ### ⚡ Dual Engine Architecture: Big Data PySpark vs Scikit-Learn Deployment

    To bridge academic Big Data requirements with real-world serverless web deployment, we engineered a dual-engine architecture:

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                Cleaned Dataset (1,980 rows)                 │
    └──────────────┬───────────────────────────────┬──────────────┘
                   │                               │
                   ▼                               ▼
    ┌─────────────────────────────┐ ┌─────────────────────────────┐
    │  PySpark MLlib Engine (Big  │ │  Scikit-Learn Engine        │
    │  Data Architecture)         │ │  (Cloud-Native Deployment)  │
    ├─────────────────────────────┤ ├─────────────────────────────┤
    │ • Tasks 121 - 125           │ │ • ColumnTransformer         │
    │ • Spark DataFrames          │ │ • StandardScaler + OneHot   │
    │ • StringIndexer + OneHot    │ │ • Joblib Model Persistence  │
    │ • VectorAssembler + Scaler  │ │ • Fast CPU Inference        │
    │ • Multi-core Distributed    │ │ • Zero Java Runtime         │
    │   Spark Logistic & RF       │ │   Requirement for Cloud     │
    │ • Binary & Multiclass       │ │ • Instant Streamlit Hosting │
    │   Classification Evaluators │ │                             │
    └─────────────────────────────┘ └─────────────────────────────┘
    ```

    #### Why Dual Engine?
    - **PySpark MLlib (Tasks 121–125):** Demonstrates distributed computing mastery, scalable feature indexing, VectorAssemblers, and distributed evaluation on cluster runtimes.
    - **Scikit-Learn Deployment:** Streamlit Community Cloud and micro-containers lack Java Virtual Machines (JVMs) and Hadoop binaries. By saving the deployable pipeline as a compact Scikit-Learn Joblib artifact (< 5 MB), the web dashboard starts in seconds and handles sub-10ms inference.
    """)

with tab4:
    st.markdown("""
    ### 🎓 Technical Viva Defense & Interview Cheat-Sheet

    #### Q1: Why did you exclude `ReviewScore` from machine learning features?
    > **Answer:** `ReviewScore` represents a textbook example of **data leakage** (target leakage). Reviews are submitted by customers days or weeks after receiving the product. If a customer intends to return an item due to dissatisfaction, they frequently leave a 1-star or 2-star review. Using post-purchase review scores at the time of checkout prediction introduces illegitimate retrospective information that would not exist in a real-time production deployment.

    #### Q2: Why is raw accuracy a dangerous evaluation metric for this dataset?
    > **Answer:** The dataset suffers from severe class imbalance: only 6.36% of orders are returned. A trivial baseline model that always predicts "NOT RETURNED" achieves **93.64% accuracy**, yet fails to detect a single returned product (0% Recall, 0% Precision, 0% F1). For return prediction, positive-class **Recall**, **Precision**, **F1-score**, and **Precision-Recall AUC (PR-AUC)** are the true indicators of business utility.

    #### Q3: How did you select Random Forest over Logistic Regression?
    > **Answer:** We conducted validation-driven model selection on a separate 15% validation split. Random Forest achieved a Validation F1-score of **0.2692** and ROC-AUC of **0.5983**, outperforming Logistic Regression (Validation F1: **0.1475**). On the final untouched test set, Random Forest confirmed superior PR-AUC (**0.2937** vs **0.1576**), effectively modeling complex non-linear interactions between categories, price points, and shipping durations.

    #### Q4: What is the purpose of decision threshold tuning?
    > **Answer:** Standard binary classifiers classify probabilities $> 0.50$ as positive. Under severe imbalance, model probabilities naturally skew downward. By tuning the cutoff $\\tau$ across the validation set (settling at $\\tau = 0.47$), we optimized the precision-recall trade-off to capture 31.6% of returns while maintaining acceptable operational precision.

    #### Q5: What are the primary limitations of the current implementation?
    > **Answer:**
    > 1. Dataset size of 2,000 records provides limited positive examples (126 returns). Scaling to 100,000+ orders will improve positive-class precision.
    > 2. Absence of textual customer review sentiment or detailed product sizing dimensions (e.g. measurements in cm).
    > 3. Static historical batch rather than continuous online streaming learning.

    #### Q6: What are the future enhancements?
    > **Answer:**
    > 1. Deep Learning tabular architectures (e.g. TabNet) and gradient boosting ensembles (XGBoost, CatBoost).
    > 2. Real-time Kafka streaming integration for real-time checkout scoring.
    > 3. Computer vision integration to cross-verify garment silhouette photographs against customer-submitted body dimensions.
    """)

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Academic Project Viva & Documentation
</div>
""", unsafe_allow_html=True)
