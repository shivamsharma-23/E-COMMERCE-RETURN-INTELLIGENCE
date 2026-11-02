"""
Page 4: Business Insights for E-Commerce Return Intelligence.

In-depth strategic analytics grounded in verified historical data patterns,
covering product categories, subcategories, shipping durations, discount elasticity,
loyalty tiers, payment risks, and concrete mitigation recommendations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card

st.set_page_config(page_title="Business Insights | Return Intelligence", page_icon="📈", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def get_insights_data():
    data_path = "dataset/Ecommerce_2000.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "Ecommerce_2000.csv")
    df = pd.read_csv(data_path)
    df_dedup = df.drop_duplicates().copy()
    df_dedup['Returned_Binary'] = (df_dedup['Returned'] == 'Yes').astype(int)
    return df_dedup


df = get_insights_data()

st.markdown(
    render_brand_header(
        "STRATEGIC BUSINESS INSIGHTS & ROOT-CAUSE ANALYTICS",
        "Empirical Evidence-Based Analytics from 1,980 Clean E-Commerce Transactions"
    ),
    unsafe_allow_html=True
)

# Executive Insights Highlights
st.markdown("### 🔍 Executive Summary: Key Findings from Empirical Data")
h1, h2, h3 = st.columns(3)

with h1:
    st.markdown("""
    <div class="kpi-card" style="border-left: 4px solid #DC2626;">
        <div class="kpi-label">Highest Return Category</div>
        <div class="kpi-value" style="color: #DC2626;">Clothing (13.04%)</div>
        <div class="kpi-sub">27 returns out of 207 orders (2.1x overall baseline)</div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown("""
    <div class="kpi-card" style="border-left: 4px solid #16A34A;">
        <div class="kpi-label">Lowest Return Category</div>
        <div class="kpi-value" style="color: #16A34A;">Grocery (1.01%)</div>
        <div class="kpi-sub">Only 2 returns out of 198 orders (perishable loyalty)</div>
    </div>
    """, unsafe_allow_html=True)

with h3:
    st.markdown("""
    <div class="kpi-card" style="border-left: 4px solid #2563EB;">
        <div class="kpi-label">Payment Disparity</div>
        <div class="kpi-value" style="color: #2563EB;">Debit Card (7.49%)</div>
        <div class="kpi-sub">vs. Gift Card (3.23%) — 2.3x return likelihood</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# SECTION 1: Product Category & Subcategory Deep Dive
st.markdown("### 1. Category & Subcategory Return Profiling")
cat_col1, cat_col2 = st.columns([1, 1])

with cat_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Product Category Hierarchy")
    cat_summary = df.groupby('ProductCategory')['Returned_Binary'].agg(
        Total_Orders='count',
        Returns='sum',
        Return_Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index().sort_values('Return_Rate', ascending=False)

    fig_cat = px.bar(
        cat_summary,
        x='Return_Rate',
        y='ProductCategory',
        orientation='h',
        color='Return_Rate',
        color_continuous_scale=[[0, '#93C5FD'], [0.5, '#2563EB'], [1, '#DC2626']],
        text=[f"{r:.1f}% ({ret}/{tot})" for r, ret, tot in zip(cat_summary['Return_Rate'], cat_summary['Returns'], cat_summary['Total_Orders'])],
        labels={'Return_Rate': 'Return Rate (%)', 'ProductCategory': ''}
    )
    fig_cat.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cat_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 👟 Top 5 vs Bottom 5 Subcategories")
    subcat_summary = df.groupby('ProductSubCategory')['Returned_Binary'].agg(
        Total_Orders='count',
        Returns='sum',
        Return_Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index()

    # Filter for subcategories with at least 15 orders to avoid single-order anomalies
    subcat_filtered = subcat_summary[subcat_summary['Total_Orders'] >= 15].copy()
    top5 = subcat_filtered.sort_values('Return_Rate', ascending=False).head(5)
    bottom5 = subcat_filtered.sort_values('Return_Rate', ascending=True).head(5)

    comp_df = pd.concat([top5, bottom5]).drop_duplicates()
    comp_df = comp_df.sort_values('Return_Rate', ascending=True)

    fig_sub = px.bar(
        comp_df,
        x='Return_Rate',
        y='ProductSubCategory',
        orientation='h',
        color='Return_Rate',
        color_continuous_scale=[[0, '#16A34A'], [0.5, '#2563EB'], [1, '#DC2626']],
        text=[f"{r:.1f}%" for r in comp_df['Return_Rate']]
    )
    fig_sub.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_sub, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Strategic Insights Box for Category
st.info("""
💡 **Actionable Category Insights:**
- **Clothing (13.0%) & Electronics (10.6%):** Apparel accounts for over 21% of all platform returns. Sizing uncertainty and "wardrobing" (buying multiple sizes with the intent to return) are dominant drivers. Implementing 3D virtual sizing widgets and detailed customer body dimension guidelines directly cuts fit-related returns.
- **Grocery (1.0%) & Books (3.1%):** Low discretionary uncertainty and non-wearable nature produce negligible return friction. These categories present high customer retention and should anchor cross-selling bundles.
""")

st.markdown("<br>", unsafe_allow_html=True)

# SECTION 2: Shipping Days & Discount Sensitivity
st.markdown("### 2. Operational Fulfillment & Pricing Elasticity")
row2_c1, row2_c2 = st.columns([1, 1])

with row2_c1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 🚚 Shipping Duration Impact on Return Propensity")
    ship_summary = df.groupby('ShippingDays')['Returned_Binary'].agg(
        Orders='count',
        Returns='sum',
        Return_Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index().sort_values('ShippingDays')

    fig_ship = px.line(
        ship_summary,
        x='ShippingDays',
        y='Return_Rate',
        markers=True,
        text=[f"{r:.1f}%" for r in ship_summary['Return_Rate']],
        labels={'ShippingDays': 'Transit Days', 'Return_Rate': 'Return Rate (%)'}
    )
    fig_ship.update_traces(line=dict(color='#2563EB', width=3), marker=dict(size=8, color='#0F172A'))
    fig_ship.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ship, use_container_width=True)
    st.caption("Orders taking 4–5 days experience peak return rates (7.2%–7.4%) due to buyer remorse accumulation.")
    st.markdown('</div>', unsafe_allow_html=True)

with row2_c2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 🏷️ Discount Percentage vs Return Behavior")
    bins = [-1, 0, 10, 20, 100]
    labels = ['0% (Full Price)', '1-10% Discount', '11-20% Discount', '>20% Heavy Discount']
    df['Disc_Band'] = pd.cut(df['DiscountPercent'], bins=bins, labels=labels)

    disc_summary = df.groupby('Disc_Band', observed=False)['Returned_Binary'].agg(
        Orders='count',
        Returns='sum',
        Return_Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index()

    fig_disc = px.bar(
        disc_summary,
        x='Disc_Band',
        y='Return_Rate',
        color='Return_Rate',
        color_continuous_scale=[[0, '#93C5FD'], [1, '#DC2626']],
        text=[f"{r:.1f}% ({o} orders)" for r, o in zip(disc_summary['Return_Rate'], disc_summary['Orders'])]
    )
    fig_disc.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_disc, use_container_width=True)
    st.caption("Heavy discounting (>20%) prompts impulse purchases that later result in higher return frequencies.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# SECTION 3: Payment Method, Device Type & Membership Loyalty
st.markdown("### 3. Customer Demographics, Payment & Channel Dynamics")
row3_c1, row3_c2, row3_c3 = st.columns(3)

with row3_c1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 💳 Payment Channels")
    pay_summary = df.groupby('PaymentMethod')['Returned_Binary'].agg(
        Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index().sort_values('Rate', ascending=False)
    
    st.dataframe(pay_summary.rename(columns={'PaymentMethod': 'Payment Method', 'Rate': 'Return Rate (%)'}), hide_index=True, use_container_width=True)
    st.caption("Debit Card and Apple Pay lead return incidence (~7.5%). Gift Card orders exhibit the lowest returns (3.2%).")
    st.markdown('</div>', unsafe_allow_html=True)

with row3_c2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 📱 Device Platforms")
    dev_summary = df.groupby('DeviceType')['Returned_Binary'].agg(
        Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index().sort_values('Rate', ascending=False)
    
    st.dataframe(dev_summary.rename(columns={'DeviceType': 'Device Type', 'Rate': 'Return Rate (%)'}), hide_index=True, use_container_width=True)
    st.caption("Desktop sessions show slightly higher return rates (6.89%) than Tablet (5.86%) and Mobile (6.31%).")
    st.markdown('</div>', unsafe_allow_html=True)

with row3_c3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 👑 Membership Tiers")
    tier_summary = df.groupby('MembershipTier')['Returned_Binary'].agg(
        Rate=lambda x: round(np.mean(x) * 100, 2)
    ).reset_index().sort_values('Rate', ascending=False)
    
    st.dataframe(tier_summary.rename(columns={'MembershipTier': 'Tier', 'Rate': 'Return Rate (%)'}), hide_index=True, use_container_width=True)
    st.caption("Gold and Bronze tiers show highest returns (~7.0%). Silver members exhibit lower return rates (5.24%).")
    st.markdown('</div>', unsafe_allow_html=True)

# Strategic Intervention Playbook Table
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("### 📋 Executive Action Playbook: Return Mitigation Strategies")
playbook_data = [
    {
        "Risk Trigger": "High Return Category (Clothing / Electronics)",
        "Root Cause": "Fit inconsistency, size bracketing, unmet spec expectations",
        "Recommended Intervention": "Embed real-time size recommendation engines; require reason confirmation on multiple-size orders; mandate spec verification videos."
    },
    {
        "Risk Trigger": "Shipping Delay (4-5 Transit Days)",
        "Root Cause": "Accumulated buyer remorse, missed event deadlines",
        "Recommended Intervention": "Implement automated SMS delivery trackers with milestone notifications; offer pre-delivery exchange options if timing shifts."
    },
    {
        "Risk Trigger": "Aggressive Promotional Discount (>20%)",
        "Root Cause": "Impulse checkout without immediate utility",
        "Recommended Intervention": "Enforce store-credit-only policies on clearance items; provide post-checkout onboarding emails reinforcing product value."
    },
    {
        "Risk Trigger": "Payment via Buy Now Pay Later / Debit",
        "Root Cause": "Lower immediate friction at checkout leads to remorse upon financial settlement",
        "Recommended Intervention": "Streamline digital return self-service while offering instant bonus store credit (e.g. +10%) for keeping or exchanging."
    }
]
st.table(pd.DataFrame(playbook_data))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Business Analytics & Strategic Action
</div>
""", unsafe_allow_html=True)
