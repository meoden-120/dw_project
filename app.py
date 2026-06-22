import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Báo Cáo Dữ Liệu Nội Bộ", 
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: #f5f7fa; }
    .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 1400px; }
    
    .report-header {
        background: #1a2744;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .report-header h1 { color: #ffffff; font-size: 18px; font-weight: 600; margin: 0; }
    .report-header .subtitle { color: #94a3b8; font-size: 12px; }
    .report-header .badge {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 3px 12px;
        border-radius: 16px;
        font-size: 11px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 12px 14px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e9f0;
        text-align: center;
    }
    .metric-value { font-size: 20px; font-weight: 700; color: #0f1724; line-height: 1.2; }
    .metric-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.3px; margin-top: 2px; }
    .metric-trend { font-size: 10px; margin-top: 2px; }
    .trend-up { color: #22c55e; }
    .trend-down { color: #ef4444; }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #0f1724;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e9f0;
    }
    
    .chart-box {
        background: #ffffff;
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid #e5e9f0;
        margin-bottom: 0.8rem;
    }
    .chart-box .chart-label { font-size: 11px; color: #64748b; font-weight: 500; margin-bottom: 4px; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e5e9f0;
        padding: 3px;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 5px 16px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] { background: #f1f5f9; color: #0f1724; }
    
    .filter-row {
        background: #ffffff;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #e5e9f0;
        margin-bottom: 1rem;
    }
    
    .dataframe { font-size: 12px; }
    .dataframe thead tr th { background: #f8fafc; color: #0f1724; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }
    
    .report-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 10px;
        padding: 0.8rem 0;
        border-top: 1px solid #e5e9f0;
        margin-top: 1rem;
    }
    
    .info-box {
        background: #f0f4ff;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        font-size: 13px;
        color: #1e3a5f;
        margin: 8px 0;
    }
    
    @media (max-width: 768px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 480px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown(f"""
<div class="report-header">
    <div>
        <h1>Báo Cáo Dữ Liệu Nội Bộ</h1>
        <div class="subtitle">Hệ thống khuyến nghị sản phẩm &amp; phân tích doanh thu</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span class="badge">SVD + SHAP</span>
        <span class="badge">{datetime.now().strftime('%d/%m/%Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    try:
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
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty:
    st.stop()

# ===================== FILTERS =====================
with st.container():
    st.markdown('<div class="filter-row">', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 1.5])
    
    with cols[0]:
        categories = ["Tất cả"] + list(df_raw["category"].dropna().unique())
        selected_category = st.selectbox("Danh mục", categories)
    
    with cols[1]:
        years = ["Tất cả"] + sorted(df_raw["year"].dropna().unique(), reverse=True)
        selected_year = st.selectbox("Năm", years)
    
    with cols[2]:
        months = ["Tất cả"] + sorted(df_raw["month"].dropna().unique())
        selected_month = st.selectbox("Tháng", months)
    
    with cols[3]:
        genders = ["Tất cả"] + list(df_raw["gender"].dropna().unique())
        selected_gender = st.selectbox("Giới tính", genders)
    
    with cols[4]:
        min_price = float(df_raw['min_price'].min())
        max_price = float(df_raw['max_price'].max())
        price_range = st.slider(
            "Khoảng giá",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5000.0,
            format="%d"
        )
    st.markdown('</div>', unsafe_allow_html=True)

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
    (df_filtered['min_price'].isna() | (df_filtered['min_price'] >= price_range[0])) &
    (df_filtered['max_price'].isna() | (df_filtered['max_price'] <= price_range[1]))
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
        <div class="metric-label">Doanh thu</div>
        <div class="metric-trend {'trend-up' if mom_growth > 0 else 'trend-down'}">
            {'↑' if mom_growth > 0 else '↓'} {abs(mom_growth):.1f}% MoM
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_qty:,.0f}</div>
        <div class="metric-label">Số lượng</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_ord:,.0f}</div>
        <div class="metric-label">Đơn hàng</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_rev:,.0f}</div>
        <div class="metric-label">Doanh thu TB</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_order_value:,.0f}</div>
        <div class="metric-label">Giá trị đơn TB</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{unique_customers:,}</div>
        <div class="metric-label">Khách hàng</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== TABS =====================
tab_overview, tab_prediction, tab_customer, tab_ai = st.tabs([
    "Tổng quan", 
    "Dự đoán & SHAP", 
    "Khách hàng", 
    "Trợ lý AI"
])

# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Doanh thu theo thời gian</div>', unsafe_allow_html=True)
        df_line = df_filtered.groupby("year_month")["revenue"].sum().reset_index().sort_values("year_month")
        if not df_line.empty:
            fig_line = px.line(df_line, x="year_month", y="revenue", markers=True)
            fig_line.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=30), xaxis_title=None, yaxis_title="Doanh thu")
            st.plotly_chart(fig_line, width='stretch')
    
    with col2:
        st.markdown('<div class="section-title">Doanh thu theo danh mục</div>', unsafe_allow_html=True)
        df_cat = df_filtered.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        if not df_cat.empty:
            fig_cat = px.bar(df_cat, x="category", y="revenue", color="category")
            fig_cat.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=30), xaxis_title=None, yaxis_title="Doanh thu", showlegend=False)
            st.plotly_chart(fig_cat, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Top 10 sản phẩm</div>', unsafe_allow_html=True)
        df_top = df_filtered.groupby("product_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)
        if not df_top.empty:
            fig_top = px.bar(df_top, x="revenue", y="product_name", orientation="h", color="revenue")
            fig_top.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Doanh thu", yaxis_title=None, coloraxis_showscale=False)
            st.plotly_chart(fig_top, width='stretch')
    
    with col2:
        st.markdown('<div class="section-title">Doanh thu theo khu vực</div>', unsafe_allow_html=True)
        df_state = df_filtered.groupby("state")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)
        if not df_state.empty:
            fig_state = px.bar(df_state, x="state", y="revenue", color="revenue")
            fig_state.update_layout(height=300, margin=dict(l=40, r=20, t=10, b=30), xaxis_title=None, yaxis_title="Doanh thu", coloraxis_showscale=False)
            st.plotly_chart(fig_state, width='stretch')
    
    with st.expander("Xem dữ liệu chi tiết", expanded=False):
        st.dataframe(df_filtered, width='stretch', height=250)

# ===================== TAB 2: PREDICTION & SHAP =====================
with tab_prediction:
    st.markdown('<div class="section-title">Dự đoán & Giải thích mô hình (SHAP)</div>', unsafe_allow_html=True)
    
    # Kiểm tra thư viện shap
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
        st.warning("⚠️ Thư viện SHAP chưa được cài đặt. Để cài đặt, chạy: pip install shap")
    
    # Kiểm tra file mô hình
    shap_file_exists = os.path.exists('shap_surrogate_model.pkl')
    
    if not shap_file_exists:
        st.markdown("""
        <div class="info-box">
            <b>📁 Chưa có mô hình SHAP</b><br>
            Để sử dụng tính năng này, bạn cần:
            <ol style="margin: 6px 0 0 20px; font-size: 13px;">
                <li>Chạy file <b>GiaiDoan3_Intelligence_NKDL_(2).ipynb</b> trên Colab</li>
                <li>Download file <b>shap_surrogate_model.pkl</b> về máy</li>
                <li>Upload file này vào cùng thư mục với app.py</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Đóng góp của các đặc trưng (SHAP Summary)</div>', unsafe_allow_html=True)
        
        if shap_available and shap_file_exists:
            try:
                import matplotlib.pyplot as plt
                
                with open('shap_surrogate_model.pkl', 'rb') as f:
                    surrogate_model = pickle.load(f)
                
                conn = duckdb.connect()
                # SỬA: Thêm product_id vào mệnh đề SELECT
                interactions = conn.execute("""
                    SELECT customer_id, product_id, SUM(TRY_CAST(Total_quantity AS DOUBLE)) AS total_qty
                    FROM 'NKDL_Project.csv'
                    WHERE product_id IS NOT NULL AND customer_id IS NOT NULL
                    GROUP BY customer_id, product_id
                """).fetchdf()
                
                dac_trung = conn.execute("""
                    SELECT customer_id, AVG(TRY_CAST(loyalty_points AS DOUBLE)) AS avg_loyalty_points,
                           SUM(TRY_CAST(Total_quantity AS DOUBLE)) AS tong_so_luong_da_mua,
                           SUM(COALESCE(TRY_CAST(Total_revenue AS DOUBLE),0)) AS tong_doanh_thu,
                           COUNT(DISTINCT product_id) AS so_san_pham_khac_nhau,
                           SUM(TRY_CAST(Total_orders AS DOUBLE)) AS tong_so_don_hang
                    FROM 'NKDL_Project.csv'
                    GROUP BY customer_id
                """).fetchdf()
                conn.close()
                
                df_surrogate = interactions.merge(dac_trung, on='customer_id', how='left')
                df_surrogate['gender'] = 'M'
                
                # Chuyển đổi dummies bình thường sau khi đã có cột product_id
                df_surrogate = pd.get_dummies(df_surrogate, columns=['product_id'])
                df_surrogate = df_surrogate.fillna(0)
                
                cols_to_drop = ['customer_id', 'total_qty']
                cols_to_drop = [c for c in cols_to_drop if c in df_surrogate.columns]
                X = df_surrogate.drop(columns=cols_to_drop)
                
                # LƯU Ý QUAN TRỌNG: 
                # Đảm bảo các cột trong X trùng khớp 100% về thứ tự và số lượng với mô hình đã train
                if hasattr(surrogate_model, "feature_names_in_"):
                    # Đồng bộ hóa các cột theo đúng model huấn luyện để tránh lỗi lệch features
                    for col in surrogate_model.feature_names_in_:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[surrogate_model.feature_names_in_]

                explainer = shap.TreeExplainer(surrogate_model)
                shap_values = explainer.shap_values(X)
                
                explainer = shap.TreeExplainer(surrogate_model)
                shap_values = explainer.shap_values(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, show=False, max_display=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.session_state['shap_values'] = shap_values
                st.session_state['X'] = X
                
            except Exception as e:
                st.error(f"Lỗi khi chạy SHAP: {e}")
        elif shap_available and not shap_file_exists:
            st.info("📁 Upload file shap_surrogate_model.pkl để xem phân tích SHAP")
        else:
            st.info("⚠️ Cần cài đặt thư viện shap: pip install shap")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Đặc trưng ảnh hưởng nhất</div>', unsafe_allow_html=True)
        
        if shap_available and shap_file_exists and 'X' in locals():
            try:
                shap_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False).head(8)
                
                fig = px.bar(shap_importance, x='importance', y='feature', orientation='h',
                            color='importance', text_auto='.3f')
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                 xaxis_title="SHAP Value", yaxis_title=None,
                                 coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.info(f"Đang tải... ({e})")
        else:
            st.info("Chưa có dữ liệu SHAP")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model performance
    st.markdown('<div class="section-title">Hiệu suất mô hình</div>', unsafe_allow_html=True)
    
    try:
        eval_log = pd.DataFrame({
            'K': [1, 3, 5, 7],
            'Precision@K': [0.1847, 0.1914, 0.1728, 0.1401],
            'Recall@K': [0.1847, 0.5742, 0.8638, 0.9808]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_eval = px.line(eval_log, x='K', y=['Precision@K', 'Recall@K'],
                              markers=True, labels={'value': 'Score', 'variable': 'Metric'})
            fig_eval.update_layout(height=220, margin=dict(l=40, r=20, t=20, b=30),
                                  legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
            fig_eval.update_yaxes(range=[0, 1.1])
            st.plotly_chart(fig_eval, width='stretch')
        
        with col2:
            st.dataframe(eval_log, width='stretch', hide_index=True)
    except Exception as e:
        st.info(f"Chưa có dữ liệu đánh giá mô hình ({e})")

# ===================== TAB 3: CUSTOMER =====================
with tab_customer:
    st.markdown('<div class="section-title">Phân tích khách hàng</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Doanh thu theo giới tính</div>', unsafe_allow_html=True)
        df_gender = df_filtered.groupby("gender")["revenue"].sum().reset_index()
        if not df_gender.empty:
            fig_gender = px.bar(df_gender, x="gender", y="revenue", color="gender")
            fig_gender.update_layout(height=250, margin=dict(l=40, r=20, t=20, b=30), xaxis_title=None, yaxis_title="Doanh thu", showlegend=False)
            st.plotly_chart(fig_gender, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Top 10 khách hàng - Điểm tích lũy</div>', unsafe_allow_html=True)
        df_loyalty = df_filtered.groupby("customer_name")["loyalty_points"].sum().reset_index().sort_values("loyalty_points", ascending=False).head(10)
        if not df_loyalty.empty:
            fig_loyalty = px.bar(df_loyalty, x="loyalty_points", y="customer_name", orientation="h", color="loyalty_points")
            fig_loyalty.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Điểm tích lũy", yaxis_title=None, coloraxis_showscale=False)
            st.plotly_chart(fig_loyalty, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Khách hàng VIP</div>', unsafe_allow_html=True)
        df_customer = df_filtered.groupby("customer_name").agg({
            "revenue": "sum", "orders": "sum", "loyalty_points": "sum"
        }).reset_index()
        if not df_customer.empty:
            df_customer["avg_order_value"] = df_customer["revenue"] / df_customer["orders"]
            df_customer = df_customer.sort_values("revenue", ascending=False).head(10)
            st.dataframe(df_customer, width='stretch', height=250, hide_index=True)
    
    with col2:
        st.markdown('<div class="section-title">ABC - Phân tích Pareto</div>', unsafe_allow_html=True)
        df_abc = df_filtered.groupby("product_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        if not df_abc.empty:
            df_abc["cumulative_pct"] = df_abc["revenue"].cumsum() / df_abc["revenue"].sum() * 100
            df_abc["category"] = "C"
            df_abc.loc[df_abc["cumulative_pct"] <= 80, "category"] = "A"
            df_abc.loc[(df_abc["cumulative_pct"] > 80) & (df_abc["cumulative_pct"] <= 95), "category"] = "B"
            
            abc_stats = df_abc.groupby("category").agg({"product_name": "count", "revenue": "sum"}).reset_index()
            abc_stats.columns = ["Hạng", "Số SP", "Doanh thu"]
            abc_stats["% Doanh thu"] = (abc_stats["Doanh thu"] / abc_stats["Doanh thu"].sum() * 100).round(1)
            
            st.dataframe(abc_stats, width='stretch', hide_index=True)
            
            fig_abc_pie = px.pie(abc_stats, values="Doanh thu", names="Hạng",
                                color_discrete_map={"A": "#22c55e", "B": "#fbbf24", "C": "#ef4444"})
            fig_abc_pie.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig_abc_pie, width='stretch')

# ===================== TAB 4: AI ASSISTANT =====================
with tab_ai:
    st.markdown('<div class="section-title">Trợ lý AI - Phân tích thông minh</div>', unsafe_allow_html=True)
    
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    
    if not GROQ_API_KEY:
        st.warning("Vui lòng cấu hình GROQ_API_KEY để sử dụng Trợ lý AI.")
    else:
        analysis_type = st.selectbox(
            "Chọn loại phân tích",
            ["Tổng quan doanh thu", "Phân tích sản phẩm", "Phân tích khách hàng", "Phân tích khu vực", "Đề xuất chiến lược"]
        )
        
        if st.button("Phân tích", width='stretch', type="primary"):
            with st.spinner("AI đang phân tích..."):
                try:
                    if analysis_type == "Tổng quan doanh thu":
                        data = df_filtered.groupby("product_name").agg({"revenue": "sum", "quantity": "sum", "orders": "sum"}).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm về doanh thu, số lượng và đơn hàng."
                    elif analysis_type == "Phân tích sản phẩm":
                        data = df_filtered.groupby(["category", "product_name"]).agg({"revenue": "sum", "quantity": "sum"}).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm theo danh mục."
                    elif analysis_type == "Phân tích khách hàng":
                        data = df_filtered.groupby(["customer_name", "gender"]).agg({"revenue": "sum", "orders": "sum", "loyalty_points": "sum"}).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 khách hàng."
                    elif analysis_type == "Phân tích khu vực":
                        data = df_filtered.groupby(["state", "city"]).agg({"revenue": "sum", "quantity": "sum", "orders": "sum"}).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 khu vực."
                    else:
                        data = df_filtered.groupby("product_name").agg({"revenue": "sum", "quantity": "sum"}).reset_index().sort_values("revenue", ascending=False).head(10)
                        prompt_instruction = "Dữ liệu top 10 sản phẩm. Hãy đề xuất chiến lược kinh doanh."
                    
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
                        model="openai/gpt-oss-20b",
                        temperature=0.3,
                        max_tokens=1024
                    )
                    
                    st.success("Phân tích hoàn tất!")
                    st.markdown(chat_completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Lỗi: {e}")

# ===================== FOOTER =====================
st.markdown(f"""
<div class="report-footer">
    Báo cáo phân tích dữ liệu nội bộ | Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
""", unsafe_allow_html=True)
