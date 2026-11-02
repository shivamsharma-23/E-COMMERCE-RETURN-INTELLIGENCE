"""
Page 2: Single Order Prediction for E-Commerce Return Intelligence.

Provides an interactive scoring form for individual orders with real-time
probability assessment, visual risk tiers, configurable decision thresholds,
and automated operational mitigation triggers.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card
from src.prediction import load_model_artifact, predict_single_order, determine_risk_tier

st.set_page_config(page_title="Single Order Prediction | Return Intelligence", page_icon="🔮", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.joblib")
    return load_model_artifact(model_path)


@st.cache_data
def get_categories_and_subcategories():
    data_path = "dataset/Ecommerce_2000.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "Ecommerce_2000.csv")
    df = pd.read_csv(data_path)
    cat_sub_map = {}
    for cat in df['ProductCategory'].dropna().unique():
        subs = sorted(df[df['ProductCategory'] == cat]['ProductSubCategory'].dropna().unique().tolist())
        cat_sub_map[cat] = subs
    states = sorted(df['State'].dropna().unique().tolist())
    cities = sorted(df['City'].dropna().unique().tolist())
    return cat_sub_map, states, cities


model_artifact = get_model()
cat_sub_map, state_list, city_list = get_categories_and_subcategories()

st.markdown(
    render_brand_header(
        "SINGLE ORDER RETURN PREDICTION",
        "Pre-Dispatch AI Risk Scoring, Probability Estimation & Targeted Mitigation Strategy"
    ),
    unsafe_allow_html=True
)

# Sidebar Configuration for Business Thresholds
with st.sidebar:
    st.markdown("### ⚙️ Risk Decision Thresholds")
    st.markdown("Adjust operational sensitivity thresholds to match your reverse logistics cost tolerance:")

    default_thresh = float(model_artifact.get('threshold', 0.47))
    custom_threshold = st.slider(
        "Decision Cutoff (Return vs Non-Return)",
        min_value=0.10,
        max_value=0.90,
        value=default_thresh,
        step=0.01,
        help="Probabilities above this threshold classify the order as RETURN."
    )

    st.markdown("---")
    st.markdown("**Risk Tier Boundaries:**")
    st.markdown("- **Low Risk:** Probability < 25%")
    st.markdown("- **Medium Risk:** 25% ≤ Probability < 50%")
    st.markdown("- **High Risk:** Probability ≥ 50%")
    st.caption("Thresholds reflect operational intervention priority, not absolute certainty.")

# Main Prediction Form Container
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("### 📝 Enter Order & Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📦 Product Specifications")
    sel_category = st.selectbox(
        "Product Category",
        options=sorted(list(cat_sub_map.keys())),
        index=sorted(list(cat_sub_map.keys())).index("Clothing") if "Clothing" in cat_sub_map else 0
    )
    
    available_subs = cat_sub_map.get(sel_category, ["Footwear", "Tops", "Bottoms"])
    sel_subcategory = st.selectbox("Product Subcategory", options=available_subs)
    
    unit_price = st.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=120.0, step=5.0)
    quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2, step=1)
    discount_pct = st.slider("Discount Applied (%)", min_value=0, max_value=70, value=15, step=5)

with col2:
    st.markdown("#### 💳 Payment & Fulfillment")
    payment_methods = ["Debit Card", "Credit Card", "Apple Pay", "Google Pay", "PayPal", "Buy Now Pay Later", "Gift Card"]
    sel_payment = st.selectbox("Payment Method", options=payment_methods, index=0)

    device_types = ["Mobile", "Desktop", "Tablet"]
    sel_device = st.selectbox("Device Type", options=device_types, index=0)

    membership_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    sel_tier = st.selectbox("Membership Tier", options=membership_tiers, index=1)

    shipping_days = st.slider("Estimated Shipping Days", min_value=1, max_value=14, value=5, step=1)

with col3:
    st.markdown("#### 📍 Geography & Timing")
    sel_state = st.selectbox("State", options=state_list, index=state_list.index("CA") if "CA" in state_list else 0)
    sel_city = st.selectbox("City", options=city_list, index=city_list.index("Sacramento") if "Sacramento" in city_list else 0)
    
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    sel_season = st.selectbox("Order Season", options=seasons, index=1)
    
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sel_dow = st.selectbox("Order Day of Week", options=days_of_week, index=0)

# Pre-computed Financial Summary Bar
total_amt = unit_price * quantity
disc_amt = total_amt * (discount_pct / 100.0)
final_amt = total_amt - disc_amt

st.markdown("---")
fin_col1, fin_col2, fin_col3 = st.columns(3)
with fin_col1:
    st.metric("Total Order Amount", f"${total_amt:,.2f}")
with fin_col2:
    st.metric("Discount Amount", f"${disc_amt:,.2f} ({discount_pct}%)")
with fin_col3:
    st.metric("Final Net Amount Paid", f"${final_amt:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🚀 PREDICT RETURN RISK", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Result Block
if predict_clicked:
    order_dict = {
        'ProductCategory': sel_category,
        'ProductSubCategory': sel_subcategory,
        'UnitPrice': unit_price,
        'Quantity': quantity,
        'TotalAmount': total_amt,
        'DiscountPercent': discount_pct,
        'DiscountAmount': disc_amt,
        'FinalAmount': final_amt,
        'PaymentMethod': sel_payment,
        'DeviceType': sel_device,
        'MembershipTier': sel_tier,
        'ShippingDays': shipping_days,
        'State': sel_state,
        'City': sel_city,
        'OrderSeason': sel_season,
        'OrderDayOfWeek': sel_dow
    }

    result = predict_single_order(
        order_dict,
        model_artifact=model_artifact,
        custom_threshold=custom_threshold
    )

    st.markdown("### 📊 Prediction Assessment Results")
    
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        # Risk Badge and Status
        if result['prediction'] == 'RETURN':
            badge_class = "risk-high" if result['probability'] >= 0.50 else "risk-medium"
            st.markdown(f'<span class="risk-badge {badge_class}">⚠️ PREDICTED STATUS: {result["prediction"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-badge risk-low">✅ PREDICTED STATUS: NOT RETURN (RETAINED)</span>', unsafe_allow_html=True)

        st.markdown(f"## **Return Probability: {result['probability_pct']}**")
        st.markdown(f"**Classification Decision:** {'Likely Return' if result['is_returned'] == 1 else 'Likely Retained'}")
        st.markdown(f"**Operational Risk Tier:** <strong style='color:{result['risk_color']}'>{result['risk_category']}</strong>", unsafe_allow_html=True)
        st.markdown(f"**Decision Cutoff Applied:** {result['threshold_used']:.2f}")

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability'] * 100,
            number={'suffix': "%", 'font': {'size': 28, 'color': '#0F172A'}},
            title={'text': "<b>Return Probability Score</b>", 'font': {'size': 14, 'color': '#64748B'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': result['risk_color'], 'thickness': 0.28},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 25], 'color': '#DCFCE7'},
                    {'range': [25, 50], 'color': '#FEF3C7'},
                    {'range': [50, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.75,
                    'value': custom_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=230,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Recommended Business Action")
        
        if result['risk_category'] == 'High Risk':
            st.error(f"🚨 **High Return Exposure Action Plan:**\n\n{result['recommended_action']}")
            st.markdown("""
            **Warehouse Directives:**
            - Attach a prepaid return label and size-exchange voucher inside packaging.
            - Send SMS: *"Need help confirming fit before delivery? Reply FIT for instant sizing agent assistance."*
            - Inspect garment stitching and zipper integrity prior to carton sealing.
            """)
        elif result['risk_category'] == 'Medium Risk':
            st.warning(f"⚠️ **Moderate Return Risk Mitigation:**\n\n{result['recommended_action']}")
            st.markdown("""
            **Engagement Tactics:**
            - Trigger post-purchase email onboarding with care guidelines and video tutorials.
            - Provide live shipping status updates to preempt transit frustration.
            """)
        else:
            st.success(f"✅ **Standard Order Fulfillment:**\n\n{result['recommended_action']}")
            st.markdown("""
            **Standard Workflow:**
            - Standard carrier routing without manual interventions.
            - Automated review invitation scheduled 5 days post delivery.
            """)
            
        st.markdown("---")
        st.caption(f"🛡️ *{result['disclaimer']}*")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Single-Order Decision Support
</div>
""", unsafe_allow_html=True)
