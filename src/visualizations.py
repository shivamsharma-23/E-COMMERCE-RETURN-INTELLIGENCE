"""
Data Visualization Module for E-Commerce Return Intelligence.

Provides enterprise SaaS-styled Plotly charts featuring professional navy/blue palettes,
clean layouts, transparent backgrounds, and responsive tooltips.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Enterprise Navy/Blue Color Palette
PRIMARY_COLOR = "#2563EB"       # Royal Blue
SECONDARY_COLOR = "#1E3A8A"     # Deep Navy
ACCENT_COLOR = "#0284C7"        # Bright Azure
ACCENT_LIGHT = "#93C5FD"        # Soft Ice Blue
ALERT_COLOR = "#DC2626"         # Crimson / Return Positive
SUCCESS_COLOR = "#16A34A"       # Forest Green / Non-Return
NEUTRAL_BG = "rgba(0,0,0,0)"
TEXT_COLOR = "#0F172A"          # Slate 900
GRID_COLOR = "#E2E8F0"          # Slate 200

BASE_LAYOUT = dict(
    paper_bgcolor=NEUTRAL_BG,
    plot_bgcolor=NEUTRAL_BG,
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color=TEXT_COLOR, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_color="#FFFFFF")
)


def plot_return_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing overall Return vs Non-Return proportion."""
    counts = df['Returned'].value_counts()
    labels = ['Not Returned (Retained)', 'Returned']
    values = [counts.get('No', 0), counts.get('Yes', 0)]
    colors = [PRIMARY_COLOR, ALERT_COLOR]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo='label+percent',
        insidetextorientation='horizontal',
        hoverinfo='label+value+percent'
    )])

    ret_rate = (values[1] / sum(values) * 100) if sum(values) > 0 else 0
    fig.update_layout(
        **BASE_LAYOUT,
        annotations=[dict(text=f"<b>{ret_rate:.1f}%</b><br>Return Rate", x=0.5, y=0.5, font_size=16, showarrow=False)],
        title=dict(text="<b>Overall Return Distribution</b>", x=0.02, y=0.96)
    )
    return fig


def plot_return_rate_by_dimension(
    df: pd.DataFrame,
    dimension_col: str,
    title: str,
    top_n: Optional[int] = None,
    horizontal: bool = False
) -> go.Figure:
    """Bar chart illustrating return rate percentage across a specific categorical dimension."""
    df_temp = df.copy()
    if 'Returned_Binary' not in df_temp.columns:
        df_temp['Returned_Binary'] = (df_temp['Returned'] == 'Yes').astype(int)

    stats = df_temp.groupby(dimension_col)['Returned_Binary'].agg(
        total_orders='count',
        returns='sum',
        return_rate=lambda x: np.mean(x) * 100
    ).reset_index()

    if top_n:
        stats = stats.sort_values('return_rate', ascending=False).head(top_n)
    else:
        stats = stats.sort_values('return_rate', ascending=True if horizontal else False)

    if horizontal:
        fig = go.Figure(go.Bar(
            y=stats[dimension_col],
            x=stats['return_rate'],
            orientation='h',
            marker=dict(
                color=stats['return_rate'],
                colorscale=[[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#1E3A8A"]],
                showscale=False
            ),
            text=[f"{r:.1f}% ({ret}/{tot})" for r, ret, tot in zip(stats['return_rate'], stats['returns'], stats['total_orders'])],
            textposition='auto'
        ))
        fig.update_layout(
            **BASE_LAYOUT,
            title=dict(text=f"<b>{title}</b>", x=0.02, y=0.96),
            xaxis=dict(title="Return Rate (%)", showgrid=True, gridcolor=GRID_COLOR),
            yaxis=dict(title="", showgrid=False)
        )
    else:
        fig = go.Figure(go.Bar(
            x=stats[dimension_col],
            y=stats['return_rate'],
            marker=dict(
                color=stats['return_rate'],
                colorscale=[[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#1E3A8A"]],
                showscale=False
            ),
            text=[f"{r:.1f}%" for r in stats['return_rate']],
            textposition='outside'
        ))
        fig.update_layout(
            **BASE_LAYOUT,
            title=dict(text=f"<b>{title}</b>", x=0.02, y=0.96),
            yaxis=dict(title="Return Rate (%)", showgrid=True, gridcolor=GRID_COLOR),
            xaxis=dict(title="", showgrid=False)
        )

    return fig


def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly order volumes and return rate trendline."""
    df_temp = df.copy()
    if 'OrderDate' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['OrderDate'], errors='coerce')
        df_temp['YearMonth'] = df_temp['Date'].dt.to_period('M').astype(str)
    else:
        return go.Figure()

    if 'Returned_Binary' not in df_temp.columns:
        df_temp['Returned_Binary'] = (df_temp['Returned'] == 'Yes').astype(int)

    monthly = df_temp.groupby('YearMonth').agg(
        total_orders=('OrderID' if 'OrderID' in df_temp.columns else 'UnitPrice', 'count'),
        returns=('Returned_Binary', 'sum'),
        return_rate=('Returned_Binary', lambda x: np.mean(x) * 100)
    ).reset_index()

    monthly = monthly.sort_values('YearMonth')

    fig = go.Figure()

    # Total Orders Bars
    fig.add_trace(go.Bar(
        x=monthly['YearMonth'],
        y=monthly['total_orders'],
        name='Total Orders',
        marker_color="#E2E8F0",
        yaxis='y'
    ))

    # Return Rate Line
    fig.add_trace(go.Scatter(
        x=monthly['YearMonth'],
        y=monthly['return_rate'],
        name='Return Rate (%)',
        mode='lines+markers',
        line=dict(color=PRIMARY_COLOR, width=3),
        marker=dict(size=7, color=ALERT_COLOR),
        yaxis='y2'
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Monthly Order Volume & Return Rate Trend</b>", x=0.02, y=0.96),
        xaxis=dict(title="Month", tickangle=-45, showgrid=False),
        yaxis=dict(title="Total Orders", showgrid=True, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Return Rate (%)", overlaying='y', side='right', showgrid=False, range=[0, max(monthly['return_rate']) * 1.5])
    )
    return fig


def plot_discount_vs_return(df: pd.DataFrame) -> go.Figure:
    """Binned analysis of discount percentage vs return rate."""
    df_temp = df.copy()
    if 'Returned_Binary' not in df_temp.columns:
        df_temp['Returned_Binary'] = (df_temp['Returned'] == 'Yes').astype(int)

    bins = [-1, 0, 10, 20, 100]
    labels = ['No Discount (0%)', 'Low (1-10%)', 'Medium (11-20%)', 'High (>20%)']
    df_temp['Discount_Tier'] = pd.cut(df_temp['DiscountPercent'], bins=bins, labels=labels)

    stats = df_temp.groupby('Discount_Tier', observed=False)['Returned_Binary'].agg(
        total='count',
        returns='sum',
        return_rate=lambda x: np.mean(x) * 100
    ).reset_index()

    fig = go.Figure(go.Bar(
        x=stats['Discount_Tier'],
        y=stats['return_rate'],
        marker_color=[SECONDARY_COLOR, ACCENT_COLOR, PRIMARY_COLOR, ALERT_COLOR],
        text=[f"{r:.1f}%<br>({int(cnt)} orders)" for r, cnt in zip(stats['return_rate'], stats['total'])],
        textposition='outside'
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Return Rate by Discount Tier</b>", x=0.02, y=0.96),
        yaxis=dict(title="Return Rate (%)", showgrid=True, gridcolor=GRID_COLOR),
        xaxis=dict(title="Discount Tier", showgrid=False)
    )
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, Any], title: str = "Confusion Matrix") -> go.Figure:
    """Confusion matrix heatmap with actual vs predicted values."""
    matrix = cm_dict.get('matrix', [[0, 0], [0, 0]])
    z = matrix
    x = ['Pred: Not Returned', 'Pred: Returned']
    y = ['Actual: Not Returned', 'Actual: Returned']

    annotations = []
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            lbl = labels[i][j]
            annotations.append(dict(
                x=x[j], y=y[i],
                text=f"<b>{val}</b><br><span style='font-size:10px;'>({lbl})</span>",
                font=dict(color="#FFFFFF" if val > np.mean(matrix) else "#0F172A", size=14),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#EFF6FF"], [0.5, "#3B82F6"], [1, "#1E3A8A"]],
        showscale=False
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"<b>{title}</b>", x=0.02, y=0.96),
        annotations=annotations,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_roc_curves(curves_dict: Dict[str, Any], lr_auc: float, rf_auc: float) -> go.Figure:
    """Comparative ROC Curve for Logistic Regression vs Random Forest."""
    fig = go.Figure()

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline (AUC = 0.50)',
        line=dict(color="#94A3B8", dash='dash')
    ))

    # Logistic Regression Curve
    lr_data = curves_dict.get('Logistic Regression', {}).get('roc_curve', {})
    if lr_data:
        fig.add_trace(go.Scatter(
            x=lr_data.get('fpr', []),
            y=lr_data.get('tpr', []),
            mode='lines',
            name=f'Logistic Regression (AUC = {lr_auc:.3f})',
            line=dict(color="#64748B", width=2)
        ))

    # Random Forest Curve
    rf_data = curves_dict.get('Random Forest', {}).get('roc_curve', {})
    if rf_data:
        fig.add_trace(go.Scatter(
            x=rf_data.get('fpr', []),
            y=rf_data.get('tpr', []),
            mode='lines',
            name=f'Random Forest (AUC = {rf_auc:.3f})',
            line=dict(color=PRIMARY_COLOR, width=3)
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Receiver Operating Characteristic (ROC) Comparison</b>", x=0.02, y=0.96),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", showgrid=True, gridcolor=GRID_COLOR, range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Recall)", showgrid=True, gridcolor=GRID_COLOR, range=[0, 1.05])
    )
    return fig


def plot_pr_curves(curves_dict: Dict[str, Any], baseline_rate: float = 0.064) -> go.Figure:
    """Precision-Recall Curves highlighting performance under minority class conditions."""
    fig = go.Figure()

    # Horizontal baseline (prevalence)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline_rate, baseline_rate],
        mode='lines',
        name=f'Prevalence Baseline ({baseline_rate * 100:.1f}%)',
        line=dict(color="#94A3B8", dash='dash')
    ))

    # Logistic Regression
    lr_pr = curves_dict.get('Logistic Regression', {}).get('pr_curve', {})
    if lr_pr:
        fig.add_trace(go.Scatter(
            x=lr_pr.get('recall', []),
            y=lr_pr.get('precision', []),
            mode='lines',
            name='Logistic Regression',
            line=dict(color="#64748B", width=2)
        ))

    # Random Forest
    rf_pr = curves_dict.get('Random Forest', {}).get('pr_curve', {})
    if rf_pr:
        fig.add_trace(go.Scatter(
            x=rf_pr.get('recall', []),
            y=rf_pr.get('precision', []),
            mode='lines',
            name='Random Forest',
            line=dict(color=PRIMARY_COLOR, width=3)
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Precision-Recall (PR) Curve Comparison</b>", x=0.02, y=0.96),
        xaxis=dict(title="Recall", showgrid=True, gridcolor=GRID_COLOR, range=[0, 1]),
        yaxis=dict(title="Precision", showgrid=True, gridcolor=GRID_COLOR, range=[0, 1.05])
    )
    return fig


def plot_feature_importance(importance_list: List[Dict[str, Any]], top_n: int = 12) -> go.Figure:
    """Horizontal bar chart showing relative predictive weight of key features."""
    top_feats = sorted(importance_list[:top_n], key=lambda x: x['importance'], reverse=False)
    names = [x['feature'].replace('num__', '').replace('cat__', '') for x in top_feats]
    scores = [x['importance'] for x in top_feats]

    fig = go.Figure(go.Bar(
        y=names,
        x=scores,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#1E3A8A"]],
            showscale=False
        ),
        text=[f"{s:.4f}" for s in scores],
        textposition='outside'
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="<b>Top Feature Importances (Random Forest)</b>", x=0.02, y=0.96),
        xaxis=dict(title="Relative Importance Score", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="", showgrid=False)
    )
    return fig
