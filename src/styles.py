"""
Enterprise UI/UX Styling and Component Helpers for Streamlit Application.
"""

# Enterprise Navy & Slate Color Palette
NAVY_PRIMARY = "#1E3A8A"
NAVY_DARK = "#0F172A"
BLUE_ACCENT = "#2563EB"
BG_LIGHT = "#F8FAFC"
CARD_BG = "#FFFFFF"
BORDER_COLOR = "#E2E8F0"
TEXT_MUTED = "#64748B"


CUSTOM_CSS = """
<style>
    /* Global Font & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    
    /* Top Header Branding */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
    }
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF;
    }
    .brand-tagline {
        font-size: 14px;
        color: #93C5FD;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* KPI Card styling */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 12px;
        color: #2563EB;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Content Cards */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.3px;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FBBF24;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
    }
    
    /* Custom Streamlit component refinements */
    div[data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    div[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
    }
</style>
"""


def render_brand_header(title: str, subtitle: str) -> str:
    """Returns HTML for top branding header."""
    return f"""
    <div class="brand-header">
        <h1 class="brand-title">📦 {title}</h1>
        <div class="brand-tagline">{subtitle}</div>
    </div>
    """


def render_kpi_card(label: str, value: str, subtext: str = "") -> str:
    """Returns HTML for an enterprise KPI card."""
    sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ''
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """
