import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle

st.set_page_config(
    page_title="Báo Cáo Phân Tích Dữ Liệu Nội Bộ", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Phong cách báo cáo nội bộ chuyên nghiệp
st.markdown("""
<style>
    /* Reset & Base */
    .main {
        background: #f8f9fc;
        padding: 0 !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        max-width: 1400px !important;
    }
    
    /* Header */
    .report-header {
        background: linear-gradient(135deg, #0f1724 0%, #1a2744 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3b82f6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .report-header h1 {
        color: #ffffff;
        font-size: 20px !important;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .report-header .subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 2px;
    }
    .report-header .badge {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 14px 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
        text-align: center;
        transition: all 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #d1d9e6;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f1724;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .metric-trend {
        font-size: 11px;
        margin-top: 3px;
    }
    .trend-up { color: #22c55e; }
    .trend-down { color: #ef4444; }
    
    /* Section Headers */
    .section-title {
        font-size: 15px !important;
        font-weight: 600;
        color: #0f1724;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #eef2f6;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title .icon {
        font-size: 18px;
    }
    
    /* Chart containers */
    .chart-box {
        background: #ffffff;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #eef2f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
        min-height: 260px;
    }
    .chart-box .chart-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 6px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #eef2f6;
        padding: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 20px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #f1f5f9 !important;
        color: #0f1724 !important;
    }
    
    /* Dataframe */
    .dataframe {
        font-size: 13px !important;
    }
    .dataframe thead tr th {
        background: #f8fafc !important;
        color: #0f1724 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .report-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 11px;
        padding: 1rem 0 0.5rem;
        border-top: 1px solid #eef2f6;
        margin-top: 1.5rem;
    }
    
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(3, 1fr); }
        .report-header { flex-direction: column; align-items: flex-start; gap: 8px; }
    }
    
    /* SHAP Waterfall */
    .shap-waterfall-container {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eef2f6;
    }
    .shap-waterfall-container .shap-title {
        font-size: 13px;
        font-weight: 600;
        color: #0f1724;
        margin-bottom: 12px;
    }
    .shap-row {
        display: flex;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13px;
    }
    .shap-row .shap-feature {
        width: 35%;
        color: #0f1724;
        font-weight: 500;
    }
    .shap-row .shap-value {
        width: 15%;
        text-align: right;
        font-weight: 600;
        padding-right: 10px;
    }
    .shap-row .shap-bar {
        flex: 1;
        height: 18px;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
    }
    .shap-row .shap-bar .bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .shap-row .shap-impact {
        width: 20%;
        font-size: 12px;
        padding-left: 10px;
        color: #64748b;
    }
    .shap-base {
        background: #f1f5f9;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        color: #0f1724;
        margin-bottom: 10px;
        text-align: center;
        font-weight: 500;
    }
    .shap-base span {
        color: #3b82f6;
    }
    .shap-prediction {
        background: #dbeafe;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        color: #1e40af;
        margin-top: 10px;
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown(f"""
<div class="report-header">
    <div>
        <h1>📊 Báo Cáo Phân Tích Dữ Liệu Nội Bộ</h1>
        <div class="subtitle">Hệ thống khuyến nghị sản phẩm &amp; phân tích doanh thu</div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span class="badge">🔵 SVD + SHAP</span>
        <span class="badge">📅 {datetime.now().strftime('%d/%m/%Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    conn = duckdb.connect()
    query = """
        SELECT 
            product_name,
            category,
            customer_name,
            customer_id,
            TRY_CAST("Year" AS INT) as year,
            TRY_CAST("Month" AS INT) as month,
            TRY_CAST("Total_revenue" AS DOUBLE) as revenue,
            TRY_CAST("Total_quantity" AS INT) as quantity,
            TRY_CAST("Total_orders" AS INT) as orders,
            TRY_CAST("loyalty_points" AS INT) as loyalty_points,
            TRY_CAST("Avg_revenue" AS DOUBLE) as avg_revenue,
            TRY_CAST("Avg_quantity" AS DOUBLE) as avg_quantity,
            TRY_CAST("Avg_loyalty_points" AS DOUBLE) as avg_loyalty_points,
            TRY_CAST("Min_price" AS DOUBLE) as min_price,
            TRY_CAST("Max_price" AS DOUBLE) as max_price,
            gender, city, state, order_date
        FROM 'NKDL_Project.csv'
    """
    df = conn.execute(query).df()
    conn.close()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    return df

try:
    df_raw = load_data()
except:
    st.error("⚠️ Không tìm thấy file dữ liệu. Vui lòng đảm bảo file 'NKDL_Project.csv' có trong thư mục.")
    st.stop()

# ===================== FILTERS =====================
with st.container():
    cols = st.columns([1, 1, 1, 1, 1.2, 0.8])
    
    with cols[0]:
        categories = ["Tất cả"] + list(df_raw["category"].dropna().unique())
        selected_category = st.selectbox("📂 Danh mục", categories, key="cat")
    
    with cols[1]:
        years = ["Tất cả"] + list(df_raw["year"].dropna().unique())
        selected_year = st.selectbox("📅 Năm", years, key="year")
    
    with cols[2]:
        months = ["Tất cả"] + list(df_raw["month"].dropna().unique())
        selected_month = st.selectbox("📆 Tháng", months, key="month")
    
    with cols[3]:
        genders = ["Tất cả"] + list(df_raw["gender"].dropna().unique())
        selected_gender = st.selectbox("👤 Giới tính", genders, key="gender")
    
    with cols[4]:
        min_price = float(df_raw['min_price'].min())
        max_price = float(df_raw['max_price'].max())
        price_range = st.slider(
            "💰 Khoảng giá",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5000.0,
            format="%d"
        )
    
    with cols[5]:
        st.markdown("<div style='margin-top: 24px;'>", unsafe_allow_html=True)
        show_all = st.button("🔄 Làm mới", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Apply filters
df_filtered = df_raw.copy()
if selected_category != "Tất cả":
    df_filtered = df_filtered[df_filtered["category"] == selected_category]
if selected_year != "Tất cả":
    df_filtered = df_filtered[df_filtered["year"] == selected_year]
if selected_month != "Tất cả":
    df_filtered = df_filtered[df_filtered["month"] == selected_month]
if selected_gender != "Tất cả":
    df_filtered = df_filtered[df_filtered["gender"] == selected_gender]
df_filtered = df_filtered[
    (df_filtered['min_price'] >= price_range[0]) & 
    (df_filtered['max_price'] <= price_range[1])
]

# ===================== METRICS =====================
total_rev = df_filtered["revenue"].sum()
total_qty = df_filtered["quantity"].sum()
total_ord = df_filtered["orders"].sum()
avg_rev = df_filtered["avg_revenue"].mean()
avg_order_value = total_rev / total_ord if total_ord > 0 else 0
unique_customers = len(df_filtered['customer_id'].unique())

# MoM growth
mom_growth = 0
unique_months = sorted(df_filtered['year_month'].unique())
if len(unique_months) >= 2:
    curr_rev = df_filtered[df_filtered['year_month'] == unique_months[-1]]['revenue'].sum()
    prev_rev = df_filtered[df_filtered['year_month'] == unique_months[-2]]['revenue'].sum()
    if prev_rev > 0:
        mom_growth = ((curr_rev - prev_rev) / prev_rev) * 100

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{total_rev:,.0f}</div>
        <div class="metric-label">Doanh Thu</div>
        <div class="metric-trend {'trend-up' if mom_growth > 0 else 'trend-down'}">
            {'↑' if mom_growth > 0 else '↓'} {abs(mom_growth):.1f}% MoM
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_qty:,.0f}</div>
        <div class="metric-label">Số Lượng</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_ord:,.0f}</div>
        <div class="metric-label">Đơn Hàng</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_rev:,.0f}</div>
        <div class="metric-label">Doanh Thu TB</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_order_value:,.0f}</div>
        <div class="metric-label">Giá Trị Đơn TB</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{unique_customers:,}</div>
        <div class="metric-label">Khách Hàng</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== TABS =====================
tab_overview, tab_prediction, tab_customer, tab_ai = st.tabs([
    "📈 Tổng Quan", 
    "🎯 Dự Đoán & SHAP", 
    "👥 Khách Hàng", 
    "🤖 Trợ Lý AI"
])

# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="section-title"><span class="icon">📊</span> Doanh Thu Theo Thời Gian</div>', unsafe_allow_html=True)
        df_line = df_filtered.groupby("year_month")["revenue"].sum().reset_index().sort_values("year_month")
        fig_line = px.line(df_line, x="year_month", y="revenue", markers=True, 
                          color_discrete_sequence=["#3b82f6"], line_shape="spline")
        fig_line.update_layout(
            height=260, margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title=None, yaxis_title="Doanh thu",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_line.update_xaxis(showgrid=False)
        fig_line.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title"><span class="icon">📊</span> Doanh Thu Theo Danh Mục</div>', unsafe_allow_html=True)
        df_cat = df_filtered.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        colors = ['#3b82f6', '#60a5fa', '#93bbfc', '#bfdbfe', '#dbeafe']
        fig_cat = px.bar(df_cat, x="category", y="revenue", color="category",
                        color_discrete_sequence=colors[:len(df_cat)])
        fig_cat.update_layout(
            height=260, margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title=None, yaxis_title="Doanh thu",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        fig_cat.update_xaxis(showgrid=False)
        fig_cat.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="section-title"><span class="icon">🏆</span> Top 10 Sản Phẩm</div>', unsafe_allow_html=True)
        df_top = df_filtered.groupby("product_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)
        fig_top = px.bar(df_top, x="revenue", y="product_name", orientation="h",
                        color="revenue", color_continuous_scale="Blues")
        fig_top.update_layout(
            height=280, margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Doanh thu", yaxis_title=None,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        fig_top.update_xaxis(gridcolor='#f1f5f9')
        fig_top.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title"><span class="icon">📍</span> Doanh Thu Theo Khu Vực</div>', unsafe_allow_html=True)
        df_state = df_filtered.groupby("state")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)
        fig_state = px.bar(df_state, x="state", y="revenue", color="revenue",
                          color_continuous_scale="Oranges")
        fig_state.update_layout(
            height=280, margin=dict(l=40, r=20, t=10, b=30),
            xaxis_title=None, yaxis_title="Doanh thu",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        fig_state.update_xaxis(showgrid=False, tickangle=15)
        fig_state.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_state, use_container_width=True)
    
    # Data table expander
    with st.expander("📋 Xem Dữ Liệu Chi Tiết", expanded=False):
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=280,
            column_config={
                "revenue": st.column_config.NumberColumn("Doanh Thu", format="%d"),
                "quantity": st.column_config.NumberColumn("Số Lượng"),
                "orders": st.column_config.NumberColumn("Đơn Hàng"),
                "loyalty_points": st.column_config.NumberColumn("Điểm TL"),
            }
        )

# ===================== TAB 2: PREDICTION & SHAP =====================
with tab_prediction:
    st.markdown('<div class="section-title"><span class="icon">🎯</span> Dự Đoán & Giải Thích Mô Hình (SHAP)</div>', unsafe_allow_html=True)
    
    # Try to load SHAP model
    try:
        with open('shap_surrogate_model.pkl', 'rb') as f:
            surrogate_model = pickle.load(f)
        shap_available = True
    except:
        shap_available = False
        st.warning("⚠️ Chưa tìm thấy mô hình SHAP. Vui lòng chạy notebook huấn luyện để tạo file 'shap_surrogate_model.pkl'")
    
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">📊 Đóng góp của các đặc trưng (SHAP Summary)</div>', unsafe_allow_html=True)
        
        if shap_available:
            try:
                import shap
                
                # Load data for SHAP
                conn = duckdb.connect()
                interactions = conn.execute("""
                    SELECT customer_id, product_id, SUM(Total_quantity) AS total_qty
                    FROM 'NKDL_Project.csv'
                    WHERE product_id IS NOT NULL AND customer_id IS NOT NULL
                    GROUP BY customer_id, product_id
                """).fetchdf()
                
                dac_trung = conn.execute("""
                    SELECT customer_id, AVG(loyalty_points) AS avg_loyalty_points,
                           SUM(Total_quantity) AS tong_so_luong_da_mua,
                           SUM(COALESCE(Total_revenue,0)) AS tong_doanh_thu,
                           COUNT(DISTINCT product_id) AS so_san_pham_khac_nhau,
                           SUM(Total_orders) AS tong_so_don_hang
                    FROM 'NKDL_Project.csv'
                    GROUP BY customer_id
                """).fetchdf()
                conn.close()
                
                df_surrogate = interactions.merge(dac_trung, on='customer_id', how='left')
                df_surrogate['gender'] = 'M'
                df_surrogate = pd.get_dummies(df_surrogate, columns=['product_id'])
                df_surrogate = df_surrogate.fillna(0)
                
                cols_to_drop = ['customer_id', 'total_qty']
                cols_to_drop = [c for c in cols_to_drop if c in df_surrogate.columns]
                X = df_surrogate.drop(columns=cols_to_drop)
                
                explainer = shap.TreeExplainer(surrogate_model)
                shap_values = explainer.shap_values(X)
                
                # Tạo SHAP summary plot với matplotlib (vẫn cần nhưng import bên trong)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, show=False, max_display=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Lưu shap_values để dùng cho waterfall
                st.session_state['shap_values'] = shap_values
                st.session_state['X'] = X
                
            except Exception as e:
                st.error(f"Lỗi khi tính SHAP: {e}")
        else:
            st.info("💡 Mô hình SHAP chưa được huấn luyện.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">📈 Đặc trưng ảnh hưởng nhất</div>', unsafe_allow_html=True)
        
        if shap_available and 'shap_values' in st.session_state:
            try:
                shap_importance = pd.DataFrame({
                    'feature': st.session_state['X'].columns,
                    'importance': np.abs(st.session_state['shap_values']).mean(axis=0)
                }).sort_values('importance', ascending=False).head(8)
                
                fig = px.bar(shap_importance, x='importance', y='feature', orientation='h',
                            color='importance', color_continuous_scale='RdYlBu_r',
                            text_auto='.3f')
                fig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="SHAP Value", yaxis_title=None,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False
                )
                fig.update_xaxis(gridcolor='#f1f5f9')
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Đang tải dữ liệu...")
        else:
            st.info("Chưa có dữ liệu SHAP")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SHAP Waterfall - Giải thích cho một khách hàng cụ thể
    if shap_available and 'shap_values' in st.session_state:
        st.markdown('<div class="section-title" style="margin-top:1rem;"><span class="icon">💡</span> Giải thích dự đoán cho khách hàng mẫu</div>', unsafe_allow_html=True)
        
        try:
            # Lấy một khách hàng mẫu
            sample_idx = 0
            sample_shap = st.session_state['shap_values'][sample_idx]
            sample_X = st.session_state['X'].iloc[sample_idx]
            
            # Tạo waterfall plot bằng plotly
            feature_names = st.session_state['X'].columns
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sample_shap
            })
            shap_df = shap_df.sort_values('shap_value', key=abs, ascending=False)
            shap_df = shap_df.head(8)  # Lấy top 8 đặc trưng
            
            # Tạo bar chart
            colors = ['#22c55e' if x > 0 else '#ef4444' for x in shap_df['shap_value']]
            fig_waterfall = px.bar(
                shap_df, 
                x='shap_value', 
                y='feature',
                orientation='h',
                color='shap_value',
                color_continuous_scale=['#ef4444', '#fbbf24', '#22c55e'],
                title="Đóng góp của từng đặc trưng vào dự đoán"
            )
            fig_waterfall.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="SHAP Value (ảnh hưởng đến dự đoán)",
                yaxis_title=None,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            fig_waterfall.update_xaxis(gridcolor='#f1f5f9')
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Giải thích ngắn
            st.markdown("""
            <div style="background: #f8fafc; padding: 12px 16px; border-radius: 8px; font-size: 13px; color: #0f1724; border-left: 3px solid #3b82f6;">
                <b>📌 Giải thích:</b> Các thanh màu xanh <span style="color:#22c55e;">⬆️</span> làm tăng khả năng được gợi ý, 
                màu đỏ <span style="color:#ef4444;">⬇️</span> làm giảm khả năng được gợi ý.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Không thể hiển thị waterfall: {e}")
    
    # Model performance
    st.markdown('<div class="section-title" style="margin-top:1rem;"><span class="icon">📊</span> Hiệu Suất Mô Hình</div>', unsafe_allow_html=True)
    
    try:
        eval_log = pd.DataFrame({
            'K': [1, 3, 5, 7],
            'Precision@K': [0.1847, 0.1914, 0.1728, 0.1401],
            'Recall@K': [0.1847, 0.5742, 0.8638, 0.9808]
        })
        
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            fig_eval = px.line(eval_log, x='K', y=['Precision@K', 'Recall@K'],
                              markers=True, labels={'value': 'Score', 'variable': 'Metric'},
                              color_discrete_map={'Precision@K': '#3b82f6', 'Recall@K': '#22c55e'})
            fig_eval.update_layout(
                height=220, margin=dict(l=40, r=20, t=20, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            fig_eval.update_xaxis(gridcolor='#f1f5f9')
            fig_eval.update_yaxis(gridcolor='#f1f5f9', range=[0, 1.1])
            st.plotly_chart(fig_eval, use_container_width=True)
        
        with col2:
            st.dataframe(eval_log, use_container_width=True, hide_index=True,
                        column_config={
                            'K': 'K',
                            'Precision@K': st.column_config.NumberColumn('Precision@K', format='%.4f'),
                            'Recall@K': st.column_config.NumberColumn('Recall@K', format='%.4f')
                        })
    except:
        st.info("Chưa có dữ liệu đánh giá mô hình")

# ===================== TAB 3: CUSTOMER =====================
with tab_customer:
    st.markdown('<div class="section-title"><span class="icon">👥</span> Phân Tích Khách Hàng</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Doanh thu theo giới tính</div>', unsafe_allow_html=True)
        df_gender = df_filtered.groupby("gender").agg({"revenue": "sum", "quantity": "sum", "orders": "sum"}).reset_index()
        fig_gender = px.bar(df_gender, x="gender", y="revenue", color="gender",
                           color_discrete_sequence=["#3b82f6", "#ec4899"])
        fig_gender.update_layout(
            height=260, margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title=None, yaxis_title="Doanh thu",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        fig_gender.update_xaxis(showgrid=False)
        fig_gender.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Top 10 khách hàng - Điểm tích lũy</div>', unsafe_allow_html=True)
        df_loyalty = df_filtered.groupby("customer_name")["loyalty_points"].sum().reset_index().sort_values("loyalty_points", ascending=False).head(10)
        fig_loyalty = px.bar(df_loyalty, x="loyalty_points", y="customer_name", orientation="h",
                            color="loyalty_points", color_continuous_scale="Oranges")
        fig_loyalty.update_layout(
            height=260, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Điểm tích lũy", yaxis_title=None,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        fig_loyalty.update_xaxis(gridcolor='#f1f5f9')
        fig_loyalty.update_yaxis(gridcolor='#f1f5f9')
        st.plotly_chart(fig_loyalty, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="section-title"><span class="icon">⭐</span> Khách Hàng VIP</div>', unsafe_allow_html=True)
        df_customer = df_filtered.groupby("customer_name").agg({
            "revenue": "sum", "orders": "sum", "loyalty_points": "sum"
        }).reset_index()
        df_customer["avg_order_value"] = df_customer["revenue"] / df_customer["orders"]
        df_customer = df_customer.sort_values("revenue", ascending=False).head(10)
        
        st.dataframe(
            df_customer,
            use_container_width=True,
            height=280,
            hide_index=True,
            column_config={
                "customer_name": "Tên KH",
                "revenue": st.column_config.NumberColumn("Doanh Thu", format="%d"),
                "orders": "Đơn Hàng",
                "loyalty_points": "Điểm TL",
                "avg_order_value": st.column_config.NumberColumn("Giá Trị TB", format="%d")
            }
        )
    
    with col2:
        st.markdown('<div class="section-title"><span class="icon">📊</span> ABC - Phân tích Pareto</div>', unsafe_allow_html=True)
        df_abc = df_filtered.groupby("product_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        df_abc["cumulative_pct"] = df_abc["revenue"].cumsum() / df_abc["revenue"].sum() * 100
        df_abc["category"] = "C"
        df_abc.loc[df_abc["cumulative_pct"] <= 80, "category"] = "A"
        df_abc.loc[(df_abc["cumulative_pct"] > 80) & (df_abc["cumulative_pct"] <= 95), "category"] = "B"
        
        abc_stats = df_abc.groupby("category").agg({
            "product_name": "count",
            "revenue": "sum"
        }).reset_index()
        abc_stats.columns = ["Hạng", "Số SP", "Doanh Thu"]
        abc_stats["% Doanh Thu"] = (abc_stats["Doanh Thu"] / abc_stats["Doanh Thu"].sum() * 100).round(1)
        
        st.dataframe(abc_stats, use_container_width=True, hide_index=True)
        
        # Pie chart for ABC
        fig_abc_pie = px.pie(abc_stats, values="Doanh Thu", names="Hạng",
                            color_discrete_map={"A": "#22c55e", "B": "#fbbf24", "C": "#ef4444"})
        fig_abc_pie.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10), showlegend=True)
        st.plotly_chart(fig_abc_pie, use_container_width=True)

# ===================== TAB 4: AI ASSISTANT =====================
with tab_ai:
    st.markdown('<div class="section-title"><span class="icon">🤖</span> Trợ Lý AI - Phân Tích Thông Minh</div>', unsafe_allow_html=True)
    
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    
    if not GROQ_API_KEY:
        st.warning("⚠️ Vui lòng cấu hình GROQ_API_KEY trong secrets để sử dụng Trợ lý AI.")
    else:
        analysis_type = st.selectbox(
            "Chọn loại phân tích",
            ["📊 Tổng quan doanh thu", "🏆 Phân tích sản phẩm", "👥 Phân tích khách hàng", "📍 Phân tích khu vực", "💡 Đề xuất chiến lược"],
            key="ai_type"
        )
        
        if st.button("🚀 Phân tích ngay", use_container_width=True, type="primary"):
            with st.spinner("🔄 AI đang phân tích dữ liệu..."):
                try:
                    # Prepare data summary
                    if analysis_type == "📊 Tổng quan doanh thu":
                        data = df_filtered.groupby("product_name").agg({
                            "revenue": "sum", 
                            "quantity": "sum", 
                            "orders": "sum"
                        }).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm về doanh thu, số lượng và đơn hàng."
                    elif analysis_type == "🏆 Phân tích sản phẩm":
                        data = df_filtered.groupby(["category", "product_name"]).agg({
                            "revenue": "sum", 
                            "quantity": "sum"
                        }).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm theo danh mục."
                    elif analysis_type == "👥 Phân tích khách hàng":
                        data = df_filtered.groupby(["customer_name", "gender"]).agg({
                            "revenue": "sum",
                            "orders": "sum",
                            "loyalty_points": "sum"
                        }).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 khách hàng."
                    elif analysis_type == "📍 Phân tích khu vực":
                        data = df_filtered.groupby(["state", "city"]).agg({
                            "revenue": "sum",
                            "quantity": "sum",
                            "orders": "sum"
                        }).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 khu vực."
                    else:  # Đề xuất chiến lược
                        data = df_filtered.groupby("product_name").agg({
                            "revenue": "sum", 
                            "quantity": "sum"
                        }).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm về doanh thu và số lượng. Hãy đề xuất chiến lược kinh doanh."
                    
                    total_revenue = df_filtered["revenue"].sum()
                    total_orders = df_filtered["orders"].sum()
                    avg_loyalty = df_filtered["avg_loyalty_points"].mean()
                    unique_customers = len(df_filtered['customer_id'].unique())
                    
                    prompt = f"""
                    {prompt_instruction}
                    
                    Dữ liệu:
                    {data.to_string(index=False)}
                    
                    Thông tin tổng quan:
                    - Tổng doanh thu: {total_revenue:,.0f} VND
                    - Tổng đơn hàng: {total_orders}
                    - Khách hàng: {unique_customers}
                    - Điểm TL TB: {avg_loyalty:.0f}
                    
                    Yêu cầu:
                    1. 3-5 nhận xét quan trọng
                    2. 3-5 đề xuất chiến lược
                    3. Cơ hội và thách thức
                    
                    Trả lời bằng tiếng Việt, sử dụng markdown.
                    """
                    
                    client = Groq(api_key=GROQ_API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0.3,
                        max_tokens=1024
                    )
                    
                    st.success("✅ Phân tích hoàn tất!")
                    st.markdown(chat_completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")

# ===================== FOOTER =====================
st.markdown(f"""
<div class="report-footer">
    Báo Cáo Phân Tích Dữ Liệu Nội Bộ | Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
""", unsafe_allow_html=True)
