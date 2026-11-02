"""
Page 1: Executive Dashboard for E-Commerce Return Intelligence.

Interactive executive monitoring with real-time filters, dynamic KPI cards,
and 8 core Plotly visualizations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.styles import CUSTOM_CSS, render_brand_header, render_kpi_card
from src.visualizations import (
    plot_return_distribution,
    plot_return_rate_by_dimension,
    plot_monthly_trend,
    plot_discount_vs_return
)

st.set_page_config(page_title="Executive Dashboard | Return Intelligence", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def get_clean_data():
    data_path = "dataset/Ecommerce_2000.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "Ecommerce_2000.csv")
    df = pd.read_csv(data_path)
    df_dedup = df.drop_duplicates().copy()
    df_dedup['Returned_Binary'] = (df_dedup['Returned'] == 'Yes').astype(int)
    return df_dedup


df_raw = get_clean_data()

st.markdown(
    render_brand_header(
        "EXECUTIVE RETURN INTELLIGENCE DASHBOARD",
        "Strategic Order Monitoring, Fulfillment Key Metrics, and Cross-Dimensional Return Analytics"
    ),
    unsafe_allow_html=True
)

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.markdown("Filter all metrics and charts in real time:")

    # Category Filter
    categories = sorted(df_raw['ProductCategory'].dropna().unique().tolist())
    sel_categories = st.multiselect("Product Category", options=categories, default=categories)

    # Payment Method Filter
    payments = sorted(df_raw['PaymentMethod'].dropna().unique().tolist())
    sel_payments = st.multiselect("Payment Method", options=payments, default=payments)

    # Membership Tier Filter
    tiers = sorted(df_raw['MembershipTier'].dropna().unique().tolist())
    sel_tiers = st.multiselect("Membership Tier", options=tiers, default=tiers)

    # Device Type Filter
    devices = sorted(df_raw['DeviceType'].dropna().unique().tolist())
    sel_devices = st.multiselect("Device Type", options=devices, default=devices)

    # Season Filter
    seasons = sorted(df_raw['OrderSeason'].dropna().unique().tolist())
    sel_seasons = st.multiselect("Order Season", options=seasons, default=seasons)

    st.markdown("---")
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

# Apply Filters
df_filtered = df_raw[
    (df_raw['ProductCategory'].isin(sel_categories)) &
    (df_raw['PaymentMethod'].isin(sel_payments)) &
    (df_raw['MembershipTier'].isin(sel_tiers)) &
    (df_raw['DeviceType'].isin(sel_devices)) &
    (df_raw['OrderSeason'].isin(sel_seasons))
].copy()

# Empty State Handling
if df_filtered.empty:
    st.warning("⚠️ No records match the selected filter combination. Please expand your filter criteria.")
    st.stop()

# Compute Executive KPIs
total_orders = len(df_filtered)
total_returns = int(df_filtered['Returned_Binary'].sum())
non_returns = total_orders - total_returns
return_rate = (total_returns / total_orders * 100) if total_orders > 0 else 0.0
total_sales = float(df_filtered['FinalAmount'].sum())
avg_order_value = total_sales / total_orders if total_orders > 0 else 0.0
avg_shipping_days = float(df_filtered['ShippingDays'].mean())

# Render 7 Required KPI Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7 = st.columns(7)

with kpi_col1:
    st.markdown(render_kpi_card("Total Orders", f"{total_orders:,}", f"{total_orders/len(df_raw)*100:.0f}% of total"), unsafe_allow_html=True)
with kpi_col2:
    st.markdown(render_kpi_card("Total Returns", f"{total_returns:,}", "Returned units"), unsafe_allow_html=True)
with kpi_col3:
    st.markdown(render_kpi_card("Non-Returned", f"{non_returns:,}", "Retained orders"), unsafe_allow_html=True)
with kpi_col4:
    st.markdown(render_kpi_card("Return Rate", f"{return_rate:.2f}%", "Cohort average"), unsafe_allow_html=True)
with kpi_col5:
    st.markdown(render_kpi_card("Avg Order Value", f"${avg_order_value:.2f}", "Net basket size"), unsafe_allow_html=True)
with kpi_col6:
    st.markdown(render_kpi_card("Total Net Sales", f"${total_sales:,.0f}", "Gross minus discounts"), unsafe_allow_html=True)
with kpi_col7:
    st.markdown(render_kpi_card("Avg Shipping", f"{avg_shipping_days:.1f} Days", "Order to doorstep"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 1 Visualizations: Return Distribution & Return Rate by Product Category
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_return_distribution(df_filtered), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with row1_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_return_rate_by_dimension(df_filtered, 'ProductCategory', "2. Return Rate by Product Category (%)", horizontal=True),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2 Visualizations: Return Rate by Payment Method & Return Rate by Membership Tier
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_return_rate_by_dimension(df_filtered, 'PaymentMethod', "3. Return Rate by Payment Method (%)"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with row2_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_return_rate_by_dimension(df_filtered, 'MembershipTier', "4. Return Rate by Membership Tier (%)"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Row 3 Visualizations: Return Rate by Device Type & Return Rate by Shipping Days
row3_col1, row3_col2 = st.columns([1, 1])

with row3_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_return_rate_by_dimension(df_filtered, 'DeviceType', "5. Return Rate by Device Type (%)"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with row3_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_return_rate_by_dimension(df_filtered, 'ShippingDays', "6. Return Rate by Shipping Days (%)"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Row 4 Visualizations: Monthly Order Trends & Discount Percentage vs Return Rate
row4_col1, row4_col2 = st.columns([1, 1])

with row4_col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_monthly_trend(df_filtered), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with row4_col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_discount_vs_return(df_filtered), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    E-Commerce Return Intelligence | Executive Analytics Suite
</div>
""", unsafe_allow_html=True)
