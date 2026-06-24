import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import pandas as pd
from datetime import datetime
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===================== CẤU HÌNH TRANG =====================
st.set_page_config(
    page_title="Báo Cáo Dữ Liệu Nội Bộ & Khuyến Nghị Sản Phẩm",
    page_icon="📊",
    layout="wide"
)

# ===================== CSS — ENTERPRISE DEEP BLUE =====================
st.markdown("""
<style>
    /* ── GOOGLE FONT ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── PALETTE ──
        --navy-950  : #050d1f  (darkest bg)
        --navy-900  : #0a1628  (sidebar / header)
        --navy-800  : #0f2040  (card bg dark)
        --navy-700  : #153058  (border / divider)
        --blue-600  : #1a56db  (primary action)
        --blue-500  : #3b82f6  (accent / links)
        --blue-400  : #60a5fa  (highlight text)
        --blue-300  : #93c5fd  (muted accent)
        --slate-100 : #f1f5f9  (page bg)
        --slate-50  : #f8fafc  (card bg light)
        --white     : #ffffff
    */

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* ── PAGE BG ── */
    .stApp {
        background: #f0f4f9;
    }
    .main .block-container {
        padding: 0 1.5rem 2rem 1.5rem;
        max-width: 1440px;
    }

    /* ══════════════════════════════════════════════
       HEADER
    ══════════════════════════════════════════════ */
    .app-header {
        background: linear-gradient(135deg, #050d1f 0%, #0a1628 60%, #0f2040 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        border-bottom: 3px solid #1a56db;
        box-shadow: 0 4px 24px rgba(5, 13, 31, 0.35);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(26,86,219,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 40%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .header-left { position: relative; z-index: 1; }
    .header-eyebrow {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 4px;
    }
    .header-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .header-sub {
        font-size: 13px;
        color: #93c5fd;
        margin-top: 5px;
        font-weight: 400;
    }
    .header-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .badge {
        background: rgba(26, 86, 219, 0.25);
        color: #93c5fd;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.35);
        letter-spacing: 0.3px;
        backdrop-filter: blur(4px);
    }
    .badge-date {
        background: rgba(255,255,255,0.07);
        color: #cbd5e1;
        border-color: rgba(255,255,255,0.12);
    }

    /* ══════════════════════════════════════════════
       FILTER BAR
    ══════════════════════════════════════════════ */
    .filter-bar {
        background: #ffffff;
        padding: 14px 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* ══════════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════════ */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 14px;
    }
    .kpi-grid-2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-bottom: 1.4rem;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.2s;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #1a56db, #3b82f6);
        border-radius: 10px 0 0 10px;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 16px rgba(26,86,219,0.10);
    }
    .kpi-label {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0a1628;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }
    .kpi-trend {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .trend-up   { color: #16a34a; }
    .trend-down { color: #dc2626; }
    .trend-icon { font-size: 14px; }

    /* ══════════════════════════════════════════════
       SECTION TITLES
    ══════════════════════════════════════════════ */
    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: #0a1628;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.2rem 0 0.75rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::before {
        content: '';
        display: inline-block;
        width: 3px;
        height: 14px;
        background: linear-gradient(180deg, #1a56db, #3b82f6);
        border-radius: 2px;
    }

    /* ══════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 4px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 7px 18px;
        border-radius: 7px;
        font-size: 12.5px;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.1px;
        transition: all 0.15s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a56db, #2563eb) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(26,86,219,0.30);
    }

    /* ══════════════════════════════════════════════
       RECOMMENDATION CARDS
    ══════════════════════════════════════════════ */
    .rec-card {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 16px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    .rec-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 16px rgba(26,86,219,0.10);
    }
    .rec-rank {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, #1a56db, #3b82f6);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; font-weight: 800; color: #ffffff;
        flex-shrink: 0;
        box-shadow: 0 3px 10px rgba(26,86,219,0.30);
    }
    .rec-body { flex: 1; }
    .rec-name {
        font-size: 15px;
        font-weight: 600;
        color: #0a1628;
        margin-bottom: 3px;
    }
    .rec-meta {
        font-size: 12px;
        color: #64748b;
    }
    .rec-score {
        background: #eff6ff;
        color: #1a56db;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid #bfdbfe;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════════════
       SHAP EXPLAIN CARD
    ══════════════════════════════════════════════ */
    .shap-card {
        background: #f0f7ff;
        border-radius: 8px;
        border-left: 4px solid #1a56db;
        padding: 12px 16px;
        margin-bottom: 6px;
    }
    .shap-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 13px;
    }
    .shap-key   { color: #1e3a5f; font-weight: 600; }
    .shap-val   { color: #334155; }
    .shap-note  { font-size: 12px; color: #475569; margin-top: 8px; padding-top: 8px; border-top: 1px solid #dbeafe; }

    /* ══════════════════════════════════════════════
       INFO / EXPLAIN BOX
    ══════════════════════════════════════════════ */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 14px;
    }
    .info-box h4 { color: #0a1628; margin: 0 0 10px 0; font-size: 14px; font-weight: 700; }
    .info-box ul { margin: 0; padding-left: 18px; color: #334155; font-size: 13.5px; line-height: 1.8; }

    /* ══════════════════════════════════════════════
       DATAFRAME OVERRIDES
    ══════════════════════════════════════════════ */
    .stDataFrame thead tr th {
        background: #0a1628 !important;
        color: #ffffff !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ══════════════════════════════════════════════
       SELECTBOX / SLIDER LABELS
    ══════════════════════════════════════════════ */
    label[data-testid="stWidgetLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ══════════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #1a56db, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 2px 8px rgba(26,86,219,0.30) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1648c0, #1d4ed8) !important;
        box-shadow: 0 4px 16px rgba(26,86,219,0.40) !important;
        transform: translateY(-1px) !important;
    }

    /* ══════════════════════════════════════════════
       EXPANDER
    ══════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #1a56db !important;
    }

    /* ══════════════════════════════════════════════
       FOOTER
    ══════════════════════════════════════════════ */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 11px;
        padding: 1rem 0 0.5rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 1.5rem;
        letter-spacing: 0.3px;
    }

    /* ══════════════════════════════════════════════
       RESPONSIVE
    ══════════════════════════════════════════════ */
    @media (max-width: 768px) {
        .kpi-grid  { grid-template-columns: repeat(2, 1fr); }
        .kpi-grid-2{ grid-template-columns: repeat(2, 1fr); }
        .header-title { font-size: 17px; }
    }
    @media (max-width: 480px) {
        .kpi-value { font-size: 20px; }
        .kpi-card  { padding: 14px 14px; }
    }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown(f"""
<div class="app-header">
    <div class="header-left">
        <div class="header-eyebrow">📊 Business Intelligence Platform</div>
        <div class="header-title">Báo Cáo Dữ Liệu Nội Bộ &amp; Khuyến Nghị Sản Phẩm</div>
        <div class="header-sub">Hệ thống phân tích doanh thu &amp; khuyến nghị sản phẩm dựa trên dữ liệu tầng Gold</div>
    </div>
    <div class="header-badges">
        <span class="badge">🧠 Mô hình SVD</span>
        <span class="badge">📈 SHAP Explainable AI</span>
        <span class="badge badge-date">{datetime.now().strftime('%d/%m/%Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    try:
        db_path = 'nkdl_warehouse.db'
        csv_path = 'NKDL_Project.csv'

        if os.path.exists(db_path):
            conn = duckdb.connect(db_path, read_only=True)
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'gold_layer'
                ORDER BY table_name
            """).fetchdf()
            if 'retail_gold' in tables['table_name'].values:
                table_name = 'retail_gold'
            elif not tables.empty:
                table_name = tables.iloc[0, 0]
            else:
                st.error("Không tìm thấy bảng dữ liệu trong database.")
                return pd.DataFrame()
            from_clause = f"gold_layer.{table_name}"
        elif os.path.exists(csv_path):
            conn = duckdb.connect()
            from_clause = f"'{csv_path}'"
        else:
            st.error("Không tìm thấy dữ liệu. Cần file nkdl_warehouse.db hoặc NKDL_Project.csv.")
            return pd.DataFrame()

        query = f"""
            SELECT
                product_id, product_name, category,
                customer_name, customer_id,
                TRY_CAST("Year" AS INT)             AS year,
                TRY_CAST("Month" AS INT)            AS month,
                TRY_CAST("Total_revenue" AS DOUBLE) AS revenue,
                TRY_CAST("Total_quantity" AS INT)   AS quantity,
                TRY_CAST("Total_orders" AS INT)     AS orders,
                TRY_CAST("loyalty_points" AS INT)   AS loyalty_points,
                TRY_CAST("Avg_revenue" AS DOUBLE)   AS avg_revenue,
                TRY_CAST("Avg_quantity" AS DOUBLE)  AS avg_quantity,
                TRY_CAST("Avg_loyalty_points" AS DOUBLE) AS avg_loyalty_points,
                TRY_CAST("Min_price" AS DOUBLE)     AS min_price,
                TRY_CAST("Max_price" AS DOUBLE)     AS max_price,
                gender, city, state, order_date
            FROM {from_clause}
        """
        df = conn.execute(query).df()
        conn.close()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty:
    st.stop()

# ===================== LOAD MODEL COMPONENTS =====================
@st.cache_resource
def load_svd_model():
    try:
        with open('svd_model_components.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("⚠️ Chưa tìm thấy file mô hình SVD. Vui lòng chạy Giai đoạn 3 để huấn luyện mô hình.")
        return None
    except Exception as e:
        st.error(f"Lỗi tải mô hình SVD: {e}")
        return None

@st.cache_resource
def load_surrogate_model():
    try:
        with open('shap_surrogate_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

svd_model = load_svd_model()
surrogate_model = load_surrogate_model()

# ===================== HÀM GỢI Ý SẢN PHẨM =====================
def get_recommendations(customer_id, k=3):
    if svd_model is None:
        return []
    try:
        customer_index = svd_model['customer_index']
        product_columns = svd_model['product_columns']
        if customer_id not in customer_index:
            return []
        idx = customer_index.index(customer_id)
        U = svd_model['U']
        sigma = svd_model['sigma']
        Vt = svd_model['Vt']
        user_means = svd_model['user_means']
        predicted = np.dot(np.dot(U[idx:idx+1], sigma), Vt) + user_means[idx]
        pred_dict = {product_columns[i]: predicted[0][i] for i in range(len(product_columns))}
        da_mua = set(df_raw[df_raw['customer_id'] == customer_id]['product_id'].dropna().unique())
        for prod in da_mua:
            if prod in pred_dict:
                pred_dict[prod] = -np.inf
        top_k = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'product_id': prod, 'score': score} for prod, score in top_k if score > -np.inf/2]
    except Exception as e:
        st.error(f"Lỗi khi sinh gợi ý: {e}")
        return []

# ===================== HÀM GIẢI THÍCH =====================
def explain_recommendation(customer_id, product_id):
    if surrogate_model is None or svd_model is None:
        return None
    try:
        customer_data = df_raw[df_raw['customer_id'] == customer_id]
        if customer_data.empty:
            return None
        features = {
            'gender': customer_data['gender'].mode().iloc[0] if not customer_data['gender'].mode().empty else 'Khong_ro',
            'avg_loyalty_points': customer_data['loyalty_points'].mean(),
            'total_quantity': customer_data['quantity'].sum(),
            'total_revenue': customer_data['revenue'].sum(),
            'unique_products': customer_data['product_id'].nunique(),
            'total_orders': customer_data['orders'].sum()
        }
        return {
            'explanation': {
                'gender': features['gender'],
                'loyalty_score': features['avg_loyalty_points'],
                'purchase_history': features['total_quantity'],
                'total_spent': features['total_revenue']
            },
            'interpretation': f"Khách hàng này có điểm tích lũy {features['avg_loyalty_points']:.0f} và đã mua {features['total_quantity']:.0f} sản phẩm trong {features['total_orders']:.0f} đơn hàng."
        }
    except Exception:
        return None

# ===================== PLOTLY THEME =====================
PLOTLY_COLORS = ["#1a56db", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#1e3a5f", "#0a1628"]
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=20, t=30, b=50),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
)

# ===================== BỘ LỌC =====================
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
cols = st.columns([1, 1, 1, 1, 1.5])

with cols[0]:
    categories = ["Tất cả"] + sorted(df_raw["category"].dropna().unique().tolist())
    selected_category = st.selectbox("Danh mục", categories)
with cols[1]:
    years = ["Tất cả"] + sorted(df_raw["year"].dropna().unique().tolist(), reverse=True)
    selected_year = st.selectbox("Năm", years)
with cols[2]:
    months = ["Tất cả"] + sorted(df_raw["month"].dropna().unique().tolist())
    selected_month = st.selectbox("Tháng", months)
with cols[3]:
    genders = ["Tất cả"] + sorted(df_raw["gender"].dropna().unique().tolist())
    selected_gender = st.selectbox("Giới tính", genders)
with cols[4]:
    min_price = float(df_raw['min_price'].min())
    max_price = float(df_raw['max_price'].max())
    price_range = st.slider("Khoảng giá", min_value=min_price, max_value=max_price,
                            value=(min_price, max_price), step=5000.0, format="%d")
st.markdown('</div>', unsafe_allow_html=True)

# ===================== ÁP DỤNG BỘ LỌC =====================
df = df_raw.copy()
if selected_category != "Tất cả": df = df[df["category"] == selected_category]
if selected_year != "Tất cả":     df = df[df["year"] == selected_year]
if selected_month != "Tất cả":    df = df[df["month"] == selected_month]
if selected_gender != "Tất cả":   df = df[df["gender"] == selected_gender]
df = df[
    (df['min_price'].isna() | (df['min_price'] >= price_range[0])) &
    (df['max_price'].isna() | (df['max_price'] <= price_range[1]))
]

# ===================== KPI =====================
total_rev        = df["revenue"].sum()
total_qty        = df["quantity"].sum()
total_ord        = df["orders"].sum()
avg_rev          = df["avg_revenue"].mean()
avg_order_value  = total_rev / total_ord if total_ord > 0 else 0
unique_customers = df['customer_id'].nunique()

mom_growth = 0
unique_months = sorted(df['year_month'].unique())
if len(unique_months) >= 2:
    curr_rev = df[df['year_month'] == unique_months[-1]]['revenue'].sum()
    prev_rev = df[df['year_month'] == unique_months[-2]]['revenue'].sum()
    if prev_rev > 0:
        mom_growth = (curr_rev - prev_rev) / prev_rev * 100

trend_cls   = "trend-up" if mom_growth >= 0 else "trend-down"
trend_arrow = "▲" if mom_growth >= 0 else "▼"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Doanh thu (VNĐ)</div>
        <div class="kpi-value">{total_rev:,.0f}</div>
        <div class="kpi-trend {trend_cls}">
            <span class="trend-icon">{trend_arrow}</span>
            {abs(mom_growth):.1f}% so tháng trước
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Số lượng bán</div>
        <div class="kpi-value">{total_qty:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Đơn hàng</div>
        <div class="kpi-value">{total_ord:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Doanh thu TB/tháng</div>
        <div class="kpi-value">{avg_rev:,.0f}</div>
    </div>
</div>
<div class="kpi-grid-2">
    <div class="kpi-card">
        <div class="kpi-label">Giá trị đơn trung bình</div>
        <div class="kpi-value">{avg_order_value:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Khách hàng</div>
        <div class="kpi-value">{unique_customers:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== TABS =====================
tab_overview, tab_customer, tab_recommend, tab_model, tab_ai = st.tabs([
    "📈 Tổng quan",
    "👥 Khách hàng",
    "🎯 Khuyến nghị sản phẩm",
    "🤖 Hiệu suất mô hình",
    "💬 Trợ lý AI"
])

# ============================================================
# TAB 1 — TỔNG QUAN
# ============================================================
with tab_overview:
    st.markdown('<div class="section-title">Doanh thu theo thời gian</div>', unsafe_allow_html=True)
    df_line = df.groupby("year_month")["revenue"].sum().reset_index().sort_values("year_month")
    if not df_line.empty:
        fig_line = px.line(df_line, x="year_month", y="revenue", markers=True,
                           labels={"year_month": "Tháng", "revenue": "Doanh thu (VNĐ)"})
        fig_line.update_traces(line=dict(color="#1a56db", width=2.5),
                               marker=dict(color="#1a56db", size=7))
        fig_line.update_layout(height=300, **PLOTLY_LAYOUT,
                               xaxis_title="Tháng", yaxis_title="Doanh thu (VNĐ)")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="section-title">Doanh thu theo danh mục</div>', unsafe_allow_html=True)
    df_cat = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    if not df_cat.empty:
        fig_cat = px.bar(df_cat, x="category", y="revenue", color="revenue",
                         labels={"category": "Danh mục", "revenue": "Doanh thu (VNĐ)"},
                         color_continuous_scale=[[0, "#93c5fd"], [1, "#1a56db"]])
        fig_cat.update_layout(height=320, **PLOTLY_LAYOUT,
                               xaxis_title="Danh mục", yaxis_title="Doanh thu (VNĐ)",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 sản phẩm theo doanh thu</div>', unsafe_allow_html=True)
    df_top = (df.groupby("product_name")["revenue"].sum()
                .reset_index().sort_values("revenue", ascending=True).tail(10))
    if not df_top.empty:
        fig_top = px.bar(df_top, x="revenue", y="product_name", orientation="h", color="revenue",
                         labels={"revenue": "Doanh thu (VNĐ)", "product_name": "Sản phẩm"},
                         color_continuous_scale=[[0, "#bfdbfe"], [1, "#1a56db"]])
        fig_top.update_layout(height=380, **PLOTLY_LAYOUT,
                               xaxis_title="Doanh thu (VNĐ)", yaxis_title=None,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 tỉnh/thành theo doanh thu</div>', unsafe_allow_html=True)
    df_state = (df.groupby("state")["revenue"].sum()
                  .reset_index().sort_values("revenue", ascending=True).tail(10))
    if not df_state.empty:
        fig_state = px.bar(df_state, x="revenue", y="state", orientation="h", color="revenue",
                           labels={"revenue": "Doanh thu (VNĐ)", "state": "Tỉnh/Thành"},
                           color_continuous_scale=[[0, "#a5f3fc"], [1, "#0891b2"]])
        fig_state.update_layout(height=360, **PLOTLY_LAYOUT,
                                 xaxis_title="Doanh thu (VNĐ)", yaxis_title=None,
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_state, use_container_width=True)

    with st.expander("📋 Xem dữ liệu chi tiết", expanded=False):
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=300)

# ============================================================
# TAB 2 — KHÁCH HÀNG
# ============================================================
with tab_customer:
    st.markdown('<div class="section-title">Doanh thu theo giới tính</div>', unsafe_allow_html=True)
    df_gender = df.groupby("gender")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    if not df_gender.empty:
        fig_gender = px.bar(df_gender, x="gender", y="revenue", color="gender",
                            labels={"gender": "Giới tính", "revenue": "Doanh thu (VNĐ)"},
                            color_discrete_sequence=["#1a56db", "#60a5fa", "#94a3b8"])
        fig_gender.update_layout(height=300, **PLOTLY_LAYOUT,
                                  xaxis_title="Giới tính", yaxis_title="Doanh thu (VNĐ)",
                                  showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 khách hàng theo điểm tích lũy</div>', unsafe_allow_html=True)
    df_loyalty = (df.groupby("customer_name")["loyalty_points"].sum()
                    .reset_index().sort_values("loyalty_points", ascending=True).tail(10))
    if not df_loyalty.empty:
        fig_loyalty = px.bar(df_loyalty, x="loyalty_points", y="customer_name", orientation="h",
                             color="loyalty_points",
                             labels={"loyalty_points": "Điểm tích lũy", "customer_name": "Khách hàng"},
                             color_continuous_scale=[[0, "#bfdbfe"], [1, "#1a56db"]])
        fig_loyalty.update_layout(height=360, **PLOTLY_LAYOUT,
                                   xaxis_title="Điểm tích lũy", yaxis_title=None,
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_loyalty, use_container_width=True)

    st.markdown('<div class="section-title">Bảng khách hàng VIP (Top 10 doanh thu)</div>', unsafe_allow_html=True)
    df_vip = df.groupby("customer_name").agg(
        Doanh_thu=("revenue", "sum"),
        Don_hang=("orders", "sum"),
        Diem_tich_luy=("loyalty_points", "sum")
    ).reset_index()
    if not df_vip.empty:
        df_vip["Gia_tri_don_TB"] = (df_vip["Doanh_thu"] / df_vip["Don_hang"]).round(0)
        df_vip = df_vip.sort_values("Doanh_thu", ascending=False).head(10).reset_index(drop=True)
        df_vip.columns = ["Khách hàng", "Doanh thu (VNĐ)", "Đơn hàng", "Điểm tích lũy", "Giá trị đơn TB"]
        st.dataframe(df_vip, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Phân tích ABC — Pareto sản phẩm</div>', unsafe_allow_html=True)
    df_abc = df.groupby("product_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    if not df_abc.empty:
        df_abc["Lũy kế (%)"] = (df_abc["revenue"].cumsum() / df_abc["revenue"].sum() * 100)
        df_abc["Nhóm"] = "C"
        df_abc.loc[df_abc["Lũy kế (%)"] <= 80, "Nhóm"] = "A"
        df_abc.loc[(df_abc["Lũy kế (%)"] > 80) & (df_abc["Lũy kế (%)"] <= 95), "Nhóm"] = "B"

        abc_stats = df_abc.groupby("Nhóm").agg(
            so_sp=("product_name", "count"),
            doanh_thu=("revenue", "sum")
        ).reset_index()
        abc_stats["% Doanh thu"] = (abc_stats["doanh_thu"] / abc_stats["doanh_thu"].sum() * 100).round(1)
        abc_stats.columns = ["Nhóm", "Số sản phẩm", "Doanh thu (VNĐ)", "% Doanh thu"]
        st.dataframe(abc_stats.sort_values("Nhóm"), use_container_width=True, hide_index=True)

        fig_abc = px.pie(abc_stats, values="Doanh thu (VNĐ)", names="Nhóm",
                         color="Nhóm",
                         color_discrete_map={"A": "#1a56db", "B": "#60a5fa", "C": "#bfdbfe"},
                         hole=0.45)
        fig_abc.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20),
                               paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_abc, use_container_width=True)

# ============================================================
# TAB 3 — KHUYẾN NGHỊ SẢN PHẨM
# ============================================================
with tab_recommend:
    st.markdown('<div class="section-title">Khuyến nghị sản phẩm thông minh (SVD)</div>', unsafe_allow_html=True)

    if svd_model is None:
        st.warning("⚠️ Mô hình SVD chưa được huấn luyện. Vui lòng chạy Giai đoạn 3 để huấn luyện mô hình.")
    else:
        customer_list = sorted(df['customer_id'].dropna().unique())
        selected_customer = st.selectbox("Chọn khách hàng để gợi ý sản phẩm", customer_list)

        if selected_customer:
            customer_name_row = df[df['customer_id'] == selected_customer]
            customer_name = customer_name_row['customer_name'].iloc[0] if not customer_name_row.empty else "Khách hàng"

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"#### 📦 Lịch sử mua hàng của <span style='color:#1a56db'>{customer_name}</span>", unsafe_allow_html=True)
                history = df[df['customer_id'] == selected_customer].groupby('product_name').agg({
                    'quantity': 'sum', 'revenue': 'sum', 'orders': 'sum'
                }).reset_index().sort_values('quantity', ascending=False)
                if not history.empty:
                    st.dataframe(
                        history.rename(columns={'product_name': 'Sản phẩm', 'quantity': 'Số lượng',
                                                'revenue': 'Doanh thu', 'orders': 'Đơn hàng'}),
                        use_container_width=True, hide_index=True, height=220
                    )
                else:
                    st.info("Khách hàng này chưa có lịch sử mua hàng.")

            with col2:
                st.markdown(f"#### 🎯 Top 3 gợi ý cho <span style='color:#1a56db'>{customer_name}</span>", unsafe_allow_html=True)
                recommendations = get_recommendations(selected_customer, k=3)

                if recommendations:
                    for idx, rec in enumerate(recommendations, 1):
                        product_id = rec['product_id']
                        score = rec['score']
                        product_info = df[df['product_id'] == product_id]
                        product_name = product_info['product_name'].iloc[0] if not product_info.empty else product_id
                        category = product_info['category'].iloc[0] if not product_info.empty else "N/A"

                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-rank">{idx}</div>
                            <div class="rec-body">
                                <div class="rec-name">{product_name}</div>
                                <div class="rec-meta">📂 {category}</div>
                            </div>
                            <div class="rec-score">⭐ {score:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        explanation = explain_recommendation(selected_customer, product_id)
                        if explanation:
                            with st.expander(f"💡 Tại sao gợi ý #{idx}?"):
                                st.markdown(f"""
                                <div class="shap-card">
                                    <div class="shap-row"><span class="shap-key">👤 Khách hàng</span><span class="shap-val">{customer_name}</span></div>
                                    <div class="shap-row"><span class="shap-key">📊 Điểm tích lũy</span><span class="shap-val">{explanation['explanation']['loyalty_score']:.0f}</span></div>
                                    <div class="shap-row"><span class="shap-key">🛒 Đã mua</span><span class="shap-val">{explanation['explanation']['purchase_history']:.0f} sản phẩm</span></div>
                                    <div class="shap-row"><span class="shap-key">💰 Tổng chi tiêu</span><span class="shap-val">{explanation['explanation']['total_spent']:,.0f} VNĐ</span></div>
                                    <div class="shap-note">📝 {explanation['interpretation']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("Không có gợi ý nào cho khách hàng này hoặc khách hàng đã mua tất cả sản phẩm.")

        st.markdown('<div class="section-title">Bảng khuyến nghị tổng hợp</div>', unsafe_allow_html=True)
        try:
            conn = duckdb.connect('nkdl_warehouse.db', read_only=True)
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'gold_layer' AND table_name = 'recommendation_results'
            """).fetchdf()
            if not tables.empty:
                df_rec = conn.execute("SELECT * FROM gold_layer.recommendation_results LIMIT 100").df()
                conn.close()
                st.dataframe(
                    df_rec.rename(columns={
                        'customer_id': 'Khách hàng', 'recommended_product_id': 'Sản phẩm gợi ý',
                        'rank': 'Hạng', 'predicted_score': 'Điểm dự đoán', 'model_name': 'Mô hình'
                    }),
                    use_container_width=True, hide_index=True, height=300
                )
            else:
                st.info("Chưa có bảng khuyến nghị tổng hợp. Hãy chạy Giai đoạn 3 để tạo bảng.")
        except Exception:
            pass

# ============================================================
# TAB 4 — HIỆU SUẤT MÔ HÌNH
# ============================================================
with tab_model:
    st.markdown('<div class="section-title">Hiệu suất mô hình gợi ý sản phẩm (SVD)</div>', unsafe_allow_html=True)

    try:
        conn = duckdb.connect('nkdl_warehouse.db', read_only=True)
        eval_log = conn.execute("""
            SELECT K, "Precision@K", "Recall@K"
            FROM gold_layer.model_evaluation_log ORDER BY K
        """).df()
        conn.close()
    except:
        eval_log = pd.DataFrame({
            'K': [1, 3, 5, 7],
            'Precision@K': [0.1847, 0.1914, 0.1728, 0.1401],
            'Recall@K':    [0.1847, 0.5742, 0.8638, 0.9808]
        })

    fig_eval = px.line(eval_log, x='K', y=['Precision@K', 'Recall@K'], markers=True,
                       labels={"value": "Điểm số", "variable": "Chỉ số", "K": "K (số gợi ý)"},
                       color_discrete_map={"Precision@K": "#1a56db", "Recall@K": "#22c55e"})
    fig_eval.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig_eval.update_layout(height=350, **PLOTLY_LAYOUT,
                           xaxis_title="K (số sản phẩm gợi ý)", yaxis_title="Điểm số",
                           yaxis_range=[0, 1.05],
                           legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    st.plotly_chart(fig_eval, use_container_width=True)

    st.markdown('<div class="section-title">Bảng kết quả đánh giá</div>', unsafe_allow_html=True)
    eval_display = eval_log.copy()
    eval_display.columns = ["K (số gợi ý)", "Precision@K", "Recall@K"]
    st.dataframe(eval_display, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
        <h4>📖 Giải thích chỉ số</h4>
        <ul>
            <li><strong>Precision@K</strong>: Trong K sản phẩm được gợi ý, tỷ lệ sản phẩm khách hàng thực sự mua.</li>
            <li><strong>Recall@K</strong>: Trong tổng số sản phẩm khách hàng đã mua, tỷ lệ được mô hình tìm đúng trong top K gợi ý.</li>
            <li><strong>Nhận xét</strong>: Precision cao nhất ở K=3, Recall tăng dần theo K — phù hợp với gợi ý top 3–5 sản phẩm.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists('shap_summary.png'):
        st.markdown('<div class="section-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
        st.image('shap_summary.png', use_container_width=True)
        st.caption("Biểu đồ SHAP summary: các đặc trưng ảnh hưởng nhiều nhất đến khuyến nghị")

    if os.path.exists('shap_waterfall_example.png'):
        st.markdown('<div class="section-title">SHAP Waterfall (Giải thích 1 gợi ý cụ thể)</div>', unsafe_allow_html=True)
        st.image('shap_waterfall_example.png', use_container_width=True)
        st.caption("Biểu đồ Waterfall: phân tích đóng góp của từng đặc trưng cho 1 khuyến nghị cụ thể")

# ============================================================
# TAB 5 — TRỢ LÝ AI
# ============================================================
with tab_ai:
    st.markdown('<div class="section-title">Trợ lý AI — Phân tích thông minh</div>', unsafe_allow_html=True)

    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        st.warning("⚠️ Vui lòng cấu hình GROQ_API_KEY trong Streamlit Secrets để sử dụng tính năng này.")
    else:
        analysis_type = st.selectbox("Chọn loại phân tích",
            ["Tổng quan doanh thu", "Phân tích sản phẩm", "Phân tích khách hàng",
             "Phân tích khu vực", "Đề xuất chiến lược"])

        if st.button("🔍 Phân tích ngay", use_container_width=True, type="primary"):
            with st.spinner("AI đang phân tích dữ liệu..."):
                try:
                    if analysis_type == "Tổng quan doanh thu":
                        data = (df.groupby("product_name")
                                  .agg({"revenue": "sum", "quantity": "sum", "orders": "sum"})
                                  .reset_index().sort_values("revenue", ascending=False).head(10))
                        prompt_instruction = "Dữ liệu top 10 sản phẩm theo doanh thu, số lượng và đơn hàng."
                    elif analysis_type == "Phân tích sản phẩm":
                        data = (df.groupby(["category", "product_name"])
                                  .agg({"revenue": "sum", "quantity": "sum"})
                                  .reset_index().sort_values("revenue", ascending=False).head(10))
                        prompt_instruction = "Dữ liệu top 10 sản phẩm theo danh mục."
                    elif analysis_type == "Phân tích khách hàng":
                        data = (df.groupby(["customer_name", "gender"])
                                  .agg({"revenue": "sum", "orders": "sum", "loyalty_points": "sum"})
                                  .reset_index().sort_values("revenue", ascending=False).head(10))
                        prompt_instruction = "Dữ liệu top 10 khách hàng theo doanh thu."
                    elif analysis_type == "Phân tích khu vực":
                        data = (df.groupby(["state", "city"])
                                  .agg({"revenue": "sum", "quantity": "sum", "orders": "sum"})
                                  .reset_index().sort_values("revenue", ascending=False).head(10))
                        prompt_instruction = "Dữ liệu top 10 khu vực theo doanh thu."
                    else:
                        data = (df.groupby("product_name")
                                  .agg({"revenue": "sum", "quantity": "sum"})
                                  .reset_index().sort_values("revenue", ascending=False).head(10))
                        prompt_instruction = "Dữ liệu top 10 sản phẩm. Đề xuất chiến lược kinh doanh."

                    prompt = f"""
{prompt_instruction}

Dữ liệu:
{data.to_string(index=False)}

Thông tin tổng quan:
- Tổng doanh thu: {total_rev:,.0f} VNĐ
- Tổng đơn hàng: {total_ord:,.0f}
- Số khách hàng: {unique_customers:,}
- Điểm tích lũy trung bình: {df["avg_loyalty_points"].mean():.0f}

Yêu cầu:
1. Đưa ra 3–5 nhận xét quan trọng từ dữ liệu
2. Đề xuất 3–5 chiến lược cải thiện
3. Nêu cơ hội và thách thức chính

Trả lời bằng tiếng Việt, dùng định dạng markdown rõ ràng.
"""
                    client = Groq(api_key=GROQ_API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=1024
                    )
                    st.success("✅ Phân tích hoàn tất!")
                    st.markdown(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Lỗi khi gọi AI: {e}")

# ===================== FOOTER =====================
st.markdown(f"""
<div class="app-footer">
    Báo cáo phân tích dữ liệu nội bộ &amp; Khuyến nghị sản phẩm &nbsp;·&nbsp;
    Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;·&nbsp;
    Powered by SVD + SHAP Explainable AI
</div>
""", unsafe_allow_html=True)