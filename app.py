
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
import shap
warnings.filterwarnings('ignore')


# ===================== CẤU HÌNH TRANG =====================
st.set_page_config(
    page_title="Báo Cáo Dữ Liệu Nội Bộ & Khuyến Nghị Sản Phẩm",
    page_icon="📊",
    layout="wide"
)

# ===================== CSS =====================
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
    .report-header h1 { color: #ffffff; font-size: 24px; font-weight: 600; margin: 0; }
    .report-header .subtitle { color: #94a3b8; font-size: 15px; }
    .report-header .badge {
        background: rgba(59,130,246,0.2);
        color: #60a5fa;
        padding: 3px 12px;
        border-radius: 16px;
        font-size: 11px;
        border: 1px solid rgba(59,130,246,0.3);
    }

    /* === METRIC GRID - HÀNG TRÊN (4 CỘT) === */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 12px;
    }
    
    /* === METRIC GRID - HÀNG DƯỚI (2 CỘT) === */
    .metric-grid-bottom {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 14px 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e5e9f0;
        text-align: center;
    }
    .metric-value { 
        font-size: 24px; 
        font-weight: 700; 
        color: #0f1724; 
        line-height: 1.3; 
    }
    .metric-label { 
        font-size: 15px; 
        color: #64748b; 
        text-transform: uppercase; 
        letter-spacing: 0.3px; 
        margin-top: 4px; 
    }
    .metric-trend { 
        font-size: 14px; 
        margin-top: 4px; 
    }
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

    .recommendation-card {
        background: #ffffff;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e5e9f0;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .recommendation-card .rank {
        font-size: 24px;
        font-weight: 700;
        color: #3b82f6;
        margin-right: 12px;
    }
    .recommendation-card .product-name {
        font-size: 16px;
        font-weight: 500;
        color: #0f1724;
    }
    .recommendation-card .product-score {
        font-size: 13px;
        color: #64748b;
    }

    .shap-card {
        background: #f8fafc;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 6px;
    }
    .shap-card .feature { font-weight: 500; color: #0f1724; }
    .shap-card .contribution { font-size: 13px; }
    .shap-positive { color: #22c55e; }
    .shap-negative { color: #ef4444; }

    .report-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 10px;
        padding: 0.8rem 0;
        border-top: 1px solid #e5e9f0;
        margin-top: 1rem;
    }

    /* === RESPONSIVE - MOBILE === */
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);  /* 2 cột trên tablet/mobile */
        }
        .metric-grid-bottom {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 480px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);  /* 2 cột trên điện thoại nhỏ */
        }
        .metric-grid-bottom {
            grid-template-columns: repeat(2, 1fr);
        }
        .metric-value { font-size: 18px; }
    }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown(f"""
<div class="report-header">
    <div>
        <h1>📊 Báo Cáo Dữ Liệu Nội Bộ & Khuyến Nghị Sản Phẩm</h1>
        <div class="subtitle">Hệ thống phân tích doanh thu & khuyến nghị sản phẩm dựa trên dữ liệu tầng Gold</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span class="badge">🧠 Mô hình SVD</span>
        <span class="badge">📈 SHAP Explainable AI</span>
        <span class="badge">{datetime.now().strftime('%d/%m/%Y')}</span>
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
                product_id,
                product_name,
                category,
                customer_name,
                customer_id,
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
            model = pickle.load(f, fix_imports=True)  # ← Thêm fix_imports=True
        return model
    except Exception:
        # Thử cách khác nếu vẫn lỗi
        try:
            import joblib
            model = joblib.load('shap_surrogate_model.pkl')
            return model
        except:
            return None

svd_model = load_svd_model()
surrogate_model = load_surrogate_model()

# svd_model = load_svd_model()
# surrogate_model = load_surrogate_model()

# # ===== DEBUG: Kiểm tra surrogate model =====
# st.sidebar.markdown("### 🔧 Debug Info")
# st.sidebar.write(f"File tồn tại: {os.path.exists('shap_surrogate_model.pkl')}")

# if os.path.exists('shap_surrogate_model.pkl'):
#     try:
#         with open('shap_surrogate_model.pkl', 'rb') as f:
#             test_model = pickle.load(f)
#         st.sidebar.success(f"✅ Load được model: {type(test_model).__name__}")
#         st.sidebar.write(f"Model type: {type(test_model)}")
#         if hasattr(test_model, 'feature_names_in_'):
#             st.sidebar.write(f"Features: {len(test_model.feature_names_in_)}")
#     except Exception as e:
#         st.sidebar.error(f"❌ Lỗi load: {str(e)}")
#         st.sidebar.error(f"❌ Exception type: {type(e).__name__}")
# else:
#     st.sidebar.error("❌ File không tồn tại")
# # =========================================

# ===================== [MODIFIED] CACHE SHAP EXPLAINER =====================
@st.cache_resource
def get_shap_explainer():
    try:
        return shap.TreeExplainer(surrogate_model)
    except Exception:
        return None

# ===================== [MODIFIED] BUILD FEATURE VECTOR =====================
def build_feature_vector(customer_id, product_id):
    customer_data = df_raw[df_raw['customer_id'] == customer_id]
    if customer_data.empty or surrogate_model is None:
        return None

    feature_names = list(getattr(surrogate_model, 'feature_names_in_', []))
    if len(feature_names) == 0:
        return None

    row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    gender = customer_data['gender'].mode().iloc[0] if not customer_data['gender'].mode().empty else 'Khong_ro'

    base_features = {
        'avg_loyalty_points': customer_data['loyalty_points'].mean(),
        'total_quantity': customer_data['quantity'].sum(),
        'total_revenue': customer_data['revenue'].sum(),
        'unique_products': customer_data['product_id'].nunique(),
        'total_orders': customer_data['orders'].sum()
    }

    for k, v in base_features.items():
        if k in row.columns:
            row.loc[0, k] = v

    gender_col = f'gender_{gender}'
    if gender_col in row.columns:
        row.loc[0, gender_col] = 1

    product_col = f'product_{product_id}'
    if product_col in row.columns:
        row.loc[0, product_col] = 1

    return row


# ===================== HÀM GỢI Ý SẢN PHẨM =====================
def get_recommendations(customer_id, k=3):
    """
    Sinh Top-K khuyến nghị cho 1 khách hàng dựa trên mô hình SVD đã huấn luyện
    """
    if svd_model is None:
        return []
    
    try:
        # Lấy danh sách customer và product từ model
        customer_index = svd_model['customer_index']
        product_columns = svd_model['product_columns']
        
        # Kiểm tra customer có trong model không
        if customer_id not in customer_index:
            return []
        
        # Lấy vị trí của customer
        idx = customer_index.index(customer_id)
        
        # Tái tạo điểm dự đoán cho customer này
        U = svd_model['U']
        sigma = svd_model['sigma']
        Vt = svd_model['Vt']
        user_means = svd_model['user_means']
        
        # Tái tạo ma trận điểm số
        predicted = np.dot(np.dot(U[idx:idx+1], sigma), Vt) + user_means[idx]
        
        # Tạo dictionary ánh xạ sản phẩm -> điểm số
        pred_dict = {product_columns[i]: predicted[0][i] for i in range(len(product_columns))}
        
        # Tìm các sản phẩm khách hàng đã mua từ dữ liệu thực tế
        da_mua = set(df_raw[df_raw['customer_id'] == customer_id]['product_id'].dropna().unique())
        
        # Loại bỏ sản phẩm đã mua và lấy top K
        for prod in da_mua:
            if prod in pred_dict:
                pred_dict[prod] = -np.inf
        
        # Sắp xếp và lấy top K
        top_k = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [{'product_id': prod, 'score': score} for prod, score in top_k if score > -np.inf/2]
    
    except Exception as e:
        st.error(f"Lỗi khi sinh gợi ý: {e}")
        return []

# ===================== HÀM GIẢI THÍCH =====================
def explain_recommendation(customer_id, product_id):
    # [MODIFIED] Sử dụng SHAP thực tế thay vì giải thích tĩnh
    if surrogate_model is None or svd_model is None:
        return None

    try:
        X = build_feature_vector(customer_id, product_id)
        if X is None:
            return None

        explainer = get_shap_explainer()
        if explainer is None:
            return None

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        tmp = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[0].values,
            'contribution': shap_values
        })

        top_df = tmp.reindex(tmp.contribution.abs().sort_values(ascending=False).index).head(3)

        customer_data = df_raw[df_raw['customer_id'] == customer_id]

        return {
            'explanation': {
                'gender': customer_data['gender'].mode().iloc[0],
                'loyalty_score': customer_data['loyalty_points'].mean(),
                'purchase_history': customer_data['quantity'].sum(),
                'total_spent': customer_data['revenue'].sum()
            },
            'interpretation': 'Giải thích bằng SHAP từ mô hình surrogate.',
            'top_features': [
                {
                    'feature': r.feature,
                    'value': r.value,
                    'contribution': r.contribution,
                    'impact': 'positive' if r.contribution >= 0 else 'negative'
                }
                for _, r in top_df.iterrows()
            ]
        }

    except Exception:
        return None

# ===================== BỘ LỌC =====================
with st.container():
    st.markdown('<div class="filter-row">', unsafe_allow_html=True)
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
        price_range = st.slider(
            "Khoảng giá",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5000.0,
            format="%d"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Áp dụng bộ lọc
df = df_raw.copy()
if selected_category != "Tất cả":
    df = df[df["category"] == selected_category]
if selected_year != "Tất cả":
    df = df[df["year"] == selected_year]
if selected_month != "Tất cả":
    df = df[df["month"] == selected_month]
if selected_gender != "Tất cả":
    df = df[df["gender"] == selected_gender]
df = df[
    (df['min_price'].isna() | (df['min_price'] >= price_range[0])) &
    (df['max_price'].isna() | (df['max_price'] <= price_range[1]))
]

# ===================== CHỈ SỐ TỔNG QUAN =====================
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

# Hàng trên: 4 chỉ số
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{total_rev:,.0f}</div>
        <div class="metric-label">Doanh thu (VNĐ)</div>
        <div class="metric-trend {'trend-up' if mom_growth >= 0 else 'trend-down'}">
            {'↑' if mom_growth >= 0 else '↓'} {abs(mom_growth):.1f}% so tháng trước
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_qty:,.0f}</div>
        <div class="metric-label">Số lượng bán</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_ord:,.0f}</div>
        <div class="metric-label">Đơn hàng</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_rev:,.0f}</div>
        <div class="metric-label">Doanh thu TB/tháng</div>
    </div>
</div>

<!-- Hàng dưới: 2 chỉ số -->
<div class="metric-grid-bottom">
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
tab_overview, tab_customer, tab_recommend, tab_model, tab_shap, tab_ai = st.tabs([
    "📈 Tổng quan",
    "👥 Khách hàng",
    "🎯 Khuyến nghị sản phẩm",
    "🤖 Hiệu suất mô hình",
    "🔍 SHAP Explainability",
    "💬 Trợ lý AI"
])

# ============================================================
# TAB 1 — TỔNG QUAN
# ============================================================
with tab_overview:

    # --- Doanh thu theo thời gian ---
    st.markdown('<div class="section-title">Doanh thu theo thời gian</div>', unsafe_allow_html=True)
    df_line = df.groupby("year_month")["revenue"].sum().reset_index().sort_values("year_month")
    if not df_line.empty:
        fig_line = px.line(
            df_line, x="year_month", y="revenue", markers=True,
            labels={"year_month": "Tháng", "revenue": "Doanh thu (VNĐ)"}
        )
        fig_line.update_traces(line_color="#3b82f6", marker_color="#1a2744")
        fig_line.update_layout(height=300, margin=dict(l=50, r=20, t=20, b=40),
                               xaxis_title="Tháng", yaxis_title="Doanh thu (VNĐ)")
        st.plotly_chart(fig_line, use_container_width=True)

    # --- Doanh thu theo danh mục ---
    st.markdown('<div class="section-title">Doanh thu theo danh mục</div>', unsafe_allow_html=True)
    df_cat = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    if not df_cat.empty:
        fig_cat = px.bar(
            df_cat, x="category", y="revenue", color="revenue",
            labels={"category": "Danh mục", "revenue": "Doanh thu (VNĐ)"},
            color_continuous_scale="Blues"
        )
        fig_cat.update_layout(height=320, margin=dict(l=50, r=20, t=20, b=60),
                               xaxis_title="Danh mục", yaxis_title="Doanh thu (VNĐ)",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # --- Top 10 sản phẩm ---
    st.markdown('<div class="section-title">Top 10 sản phẩm theo doanh thu</div>', unsafe_allow_html=True)
    df_top = (df.groupby("product_name")["revenue"].sum()
                .reset_index().sort_values("revenue", ascending=True).tail(10))
    if not df_top.empty:
        fig_top = px.bar(
            df_top, x="revenue", y="product_name", orientation="h", color="revenue",
            labels={"revenue": "Doanh thu (VNĐ)", "product_name": "Sản phẩm"},
            color_continuous_scale="Blues"
        )
        fig_top.update_layout(height=380, margin=dict(l=20, r=20, t=10, b=40),
                               xaxis_title="Doanh thu (VNĐ)", yaxis_title=None,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    # --- Doanh thu theo tỉnh/thành ---
    st.markdown('<div class="section-title">Top 10 tỉnh/thành theo doanh thu</div>', unsafe_allow_html=True)
    df_state = (df.groupby("state")["revenue"].sum()
                  .reset_index().sort_values("revenue", ascending=True).tail(10))
    if not df_state.empty:
        fig_state = px.bar(
            df_state, x="revenue", y="state", orientation="h", color="revenue",
            labels={"revenue": "Doanh thu (VNĐ)", "state": "Tỉnh/Thành"},
            color_continuous_scale="Teal"
        )
        fig_state.update_layout(height=360, margin=dict(l=20, r=20, t=10, b=40),
                                 xaxis_title="Doanh thu (VNĐ)", yaxis_title=None,
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_state, use_container_width=True)

    # --- Bảng dữ liệu chi tiết ---
    with st.expander("📋 Xem dữ liệu chi tiết", expanded=False):
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=300)

# ============================================================
# TAB 2 — KHÁCH HÀNG
# ============================================================
with tab_customer:

    # --- Doanh thu theo giới tính ---
    st.markdown('<div class="section-title">Doanh thu theo giới tính</div>', unsafe_allow_html=True)
    df_gender = df.groupby("gender")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    if not df_gender.empty:
        fig_gender = px.bar(
            df_gender, x="gender", y="revenue", color="gender",
            labels={"gender": "Giới tính", "revenue": "Doanh thu (VNĐ)"},
            color_discrete_sequence=["#3b82f6", "#f472b6", "#94a3b8"]
        )
        fig_gender.update_layout(height=300, margin=dict(l=50, r=20, t=20, b=50),
                                  xaxis_title="Giới tính", yaxis_title="Doanh thu (VNĐ)",
                                  showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)

    # --- Top 10 khách hàng theo điểm tích lũy ---
    st.markdown('<div class="section-title">Top 10 khách hàng theo điểm tích lũy</div>', unsafe_allow_html=True)
    df_loyalty = (df.groupby("customer_name")["loyalty_points"].sum()
                    .reset_index().sort_values("loyalty_points", ascending=True).tail(10))
    if not df_loyalty.empty:
        fig_loyalty = px.bar(
            df_loyalty, x="loyalty_points", y="customer_name", orientation="h",
            color="loyalty_points",
            labels={"loyalty_points": "Điểm tích lũy", "customer_name": "Khách hàng"},
            color_continuous_scale="Oranges"
        )
        fig_loyalty.update_layout(height=360, margin=dict(l=20, r=20, t=10, b=40),
                                   xaxis_title="Điểm tích lũy", yaxis_title=None,
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_loyalty, use_container_width=True)

    # --- Bảng khách hàng VIP ---
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

    # --- Phân tích ABC / Pareto ---
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

        fig_abc = px.pie(
            abc_stats, values="Doanh thu (VNĐ)", names="Nhóm",
            color="Nhóm",
            color_discrete_map={"A": "#22c55e", "B": "#fbbf24", "C": "#ef4444"},
            hole=0.4
        )
        fig_abc.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_abc, use_container_width=True)

# ============================================================
# TAB 3 — KHUYẾN NGHỊ SẢN PHẨM (MỚI - TÍCH HỢP TỪ GĐ3)
# ============================================================
with tab_recommend:
    st.markdown('<div class="section-title">🎯 Khuyến nghị sản phẩm thông minh (SVD)</div>', unsafe_allow_html=True)

    if svd_model is None:
        st.warning("⚠️ Mô hình SVD chưa được huấn luyện. Vui lòng chạy Giai đoạn 3 để huấn luyện mô hình.")
    else:
        # Chọn khách hàng
        customer_list = sorted(df['customer_id'].dropna().unique())
        selected_customer = st.selectbox("Chọn khách hàng để gợi ý sản phẩm", customer_list)

        if selected_customer:
            # Tìm tên khách hàng
            customer_name = df[df['customer_id'] == selected_customer]['customer_name'].iloc[0] if not df[df['customer_id'] == selected_customer].empty else "Khách hàng"

            col1, col2 = st.columns([1, 1])

            with col1:
                # Lịch sử mua hàng của khách hàng
                st.markdown(f"#### 📦 Lịch sử mua hàng của <span style='color:#3b82f6;'>{customer_name}</span>", unsafe_allow_html=True)
                history = df[df['customer_id'] == selected_customer].groupby('product_name').agg({
                    'quantity': 'sum',
                    'revenue': 'sum',
                    'orders': 'sum'
                }).reset_index().sort_values('quantity', ascending=False)

                if not history.empty:
                    st.dataframe(
                        history.rename(columns={'product_name': 'Sản phẩm', 'quantity': 'Số lượng', 'revenue': 'Doanh thu', 'orders': 'Đơn hàng'}),
                        use_container_width=True,
                        hide_index=True,
                        height=200
                    )
                else:
                    st.info("Khách hàng này chưa có lịch sử mua hàng.")

            with col2:
                # Gợi ý sản phẩm
                st.markdown(f"#### 🎯 Top 3 gợi ý cho <span style='color:#3b82f6;'>{customer_name}</span>", unsafe_allow_html=True)

                recommendations = get_recommendations(selected_customer, k=3)

                if recommendations:
                    for idx, rec in enumerate(recommendations, 1):
                        product_id = rec['product_id']
                        score = rec['score']

                        # Tìm tên sản phẩm
                        product_info = df[df['product_id'] == product_id]
                        product_name = product_info['product_name'].iloc[0] if not product_info.empty else product_id

                        # Tìm category
                        category = product_info['category'].iloc[0] if not product_info.empty else "N/A"

                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span class="rank">#{idx}</span>
                                <div>
                                    <div class="product-name">{product_name}</div>
                                    <div class="product-score">📂 {category} | Điểm tin cậy: {score:.3f}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Giải thích SHAP (nếu có)
                        explanation = explain_recommendation(selected_customer, product_id)
                        if explanation:
                            with st.expander(f"💡 Giải thích tại sao gợi ý #{idx} này?"):
                                st.markdown(f"""
                                <div class="shap-card">
                                    <div><span class="feature">👤 Khách hàng:</span> {customer_name}</div>
                                    <div><span class="feature">📊 Điểm tích lũy:</span> {explanation['explanation']['loyalty_score']:.0f}</div>
                                    <div><span class="feature">🛒 Đã mua:</span> {explanation['explanation']['purchase_history']:.0f} sản phẩm</div>
                                    <div><span class="feature">💰 Tổng chi tiêu:</span> {explanation['explanation']['total_spent']:,.0f} VNĐ</div>
                                    <div style="margin-top:8px; padding-top:8px; border-top:1px solid #e5e9f0;">
                                        <span style="color:#64748b; font-size:13px;">📝 {explanation['interpretation']}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("Không có gợi ý nào cho khách hàng này hoặc khách hàng đã mua tất cả sản phẩm.")

        # Bảng khuyến nghị cho toàn bộ khách hàng (lấy từ warehouse nếu có)
        st.markdown('<div class="section-title" style="margin-top:20px;">📋 Bảng khuyến nghị tổng hợp</div>', unsafe_allow_html=True)

        try:
            # Kiểm tra xem bảng recommendation_results có tồn tại không
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
                        'customer_id': 'Khách hàng',
                        'recommended_product_id': 'Sản phẩm gợi ý',
                        'rank': 'Hạng',
                        'predicted_score': 'Điểm dự đoán',
                        'model_name': 'Mô hình'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )
            else:
                st.info("Chưa có bảng khuyến nghị tổng hợp. Hãy chạy Giai đoạn 3 để tạo bảng.")
        except Exception as e:
            pass

# ============================================================
# TAB 4 — HIỆU SUẤT MÔ HÌNH SVD
# ============================================================
with tab_model:
    st.markdown('<div class="section-title">🤖 Hiệu suất mô hình gợi ý sản phẩm (SVD)</div>', unsafe_allow_html=True)

    # Đọc log đánh giá từ warehouse nếu có
    try:
        conn = duckdb.connect('nkdl_warehouse.db', read_only=True)
        eval_log = conn.execute("""
            SELECT K, "Precision@K", "Recall@K"
            FROM gold_layer.model_evaluation_log
            ORDER BY K
        """).df()
        conn.close()
    except:
        # Fallback: dùng dữ liệu mẫu từ GĐ3
        eval_log = pd.DataFrame({
            'K': [1, 3, 5, 7],
            'Precision@K': [0.1847, 0.1914, 0.1728, 0.1401],
            'Recall@K': [0.1847, 0.5742, 0.8638, 0.9808]
        })

    # Biểu đồ đường Precision & Recall
    fig_eval = px.line(
        eval_log, x='K', y=['Precision@K', 'Recall@K'],
        markers=True,
        labels={"value": "Điểm số", "variable": "Chỉ số", "K": "K (số gợi ý)"},
        color_discrete_map={"Precision@K": "#3b82f6", "Recall@K": "#22c55e"}
    )
    fig_eval.update_layout(
        height=350, margin=dict(l=50, r=20, t=30, b=50),
        xaxis_title="K (số sản phẩm gợi ý)",
        yaxis_title="Điểm số",
        yaxis_range=[0, 1.05],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Bảng chi tiết
    st.markdown('<div class="section-title">Bảng kết quả đánh giá</div>', unsafe_allow_html=True)
    eval_display = eval_log.copy()
    eval_display.columns = ["K (số gợi ý)", "Precision@K", "Recall@K"]
    st.dataframe(eval_display, use_container_width=True, hide_index=True)

    # Thông tin giải thích
    st.markdown("""
    <div style="background:#f0f9ff; padding:16px; border-radius:8px; border:1px solid #bae6fd; margin-top:16px;">
        <h4 style="margin:0 0 8px 0; color:#0f1724;">📖 Giải thích chỉ số</h4>
        <ul style="margin:0; padding-left:20px; color:#334155; font-size:14px;">
            <li><strong>Precision@K</strong>: Trong K sản phẩm được gợi ý, tỷ lệ sản phẩm khách hàng thực sự mua.</li>
            <li><strong>Recall@K</strong>: Trong tổng số sản phẩm khách hàng đã mua, tỷ lệ được mô hình tìm đúng trong top K gợi ý.</li>
            <li><strong>Nhận xét</strong>: Precision cao nhất ở K=3, Recall tăng dần theo K — phù hợp với gợi ý top 3–5 sản phẩm.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # SHAP Summary (nếu có ảnh)
    if os.path.exists('shap_summary.png'):
        st.markdown('<div class="section-title">📊 SHAP Feature Importance</div>', unsafe_allow_html=True)
        st.image('shap_summary.png', use_container_width=True)
        st.caption("Biểu đồ SHAP summary: các đặc trưng ảnh hưởng nhiều nhất đến khuyến nghị")

    if os.path.exists('shap_waterfall_example.png'):
        st.markdown('<div class="section-title">🌊 SHAP Waterfall (Giải thích 1 gợi ý cụ thể)</div>', unsafe_allow_html=True)
        st.image('shap_waterfall_example.png', use_container_width=True)
        st.caption("Biểu đồ Waterfall: phân tích đóng góp của từng đặc trưng cho 1 khuyến nghị cụ thể")

# ============================================================
# TAB SHAP — GIẢI THÍCH MÔ HÌNH
# ============================================================
with tab_shap:
    st.markdown('<div class="section-title">🔍 SHAP — Giải thích khuyến nghị sản phẩm</div>', unsafe_allow_html=True)
    
    if surrogate_model is None:
        st.info("Mô hình SHAP chưa được tải. Cần file shap_surrogate_model.pkl để hiển thị giải thích.")
    else:
        st.success("Mô hình SHAP đã sẵn sàng")
        
        # Chọn khách hàng để giải thích
        customer_list = sorted(df['customer_id'].dropna().unique())
        selected_customer_shap = st.selectbox(
            "Chọn khách hàng để xem giải thích SHAP",
            customer_list,
            key="shap_customer_select"
        )
        
        if selected_customer_shap:
            # Lấy khuyến nghị cho khách hàng
            recommendations = get_recommendations(selected_customer_shap, k=5)
            
            if recommendations:
                st.markdown(f"#### 📊 Giải thích SHAP cho {len(recommendations)} khuyến nghị hàng đầu")
                
                for idx, rec in enumerate(recommendations, 1):
                    product_id = rec['product_id']
                    score = rec['score']
                    
                    # Tìm tên sản phẩm
                    product_info = df[df['product_id'] == product_id]
                    product_name = product_info['product_name'].iloc[0] if not product_info.empty else product_id
                    
                    st.markdown(f"**#{idx}: {product_name}** (Điểm: {score:.3f})")
                    
                    # Giải thích SHAP
                    explanation = explain_recommendation(selected_customer_shap, product_id)
                    if explanation:
                        cols_shap = st.columns(len(explanation['top_features']))
                        
                        for i, feature in enumerate(explanation['top_features']):
                            with cols_shap[i]:
                                color = "#22c55e" if feature['impact'] == 'positive' else "#ef4444"
                                icon = "▲" if feature['impact'] == 'positive' else "▼"
                                
                                st.markdown(f"""
                                <div style="background:#ffffff; padding:12px; border-radius:8px; 
                                            border:1px solid #e5e9f0; text-align:center; margin:4px 0;">
                                    <div style="font-size:12px; color:#64748b; margin-bottom:4px;">
                                        {feature['feature']}
                                    </div>
                                    <div style="font-size:20px; font-weight:700; color:{color};">
                                        {icon} {abs(feature['contribution']):.3f}
                                    </div>
                                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">
                                        Giá trị: {feature['value']:.2f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Thông tin khách hàng
                        with st.expander(f"📋 Thông tin chi tiết khách hàng"):
                            st.markdown(f"""
                            - **Giới tính**: {explanation['explanation']['gender']}
                            - **Điểm tích lũy TB**: {explanation['explanation']['loyalty_score']:.0f}
                            - **Tổng sản phẩm đã mua**: {explanation['explanation']['purchase_history']:.0f}
                            - **Tổng chi tiêu**: {explanation['explanation']['total_spent']:,.0f} VNĐ
                            """)
                    else:
                        st.info(f"Không thể tạo giải thích SHAP cho sản phẩm này")
                    st.markdown("---")
            else:
                st.info("Không có khuyến nghị nào cho khách hàng này.")
    
    # Hiển thị SHAP summary plots
    st.markdown('<div class="section-title" style="margin-top:20px;">📈 Tổng quan SHAP</div>', unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if os.path.exists('shap_summary.png'):
            st.image('shap_summary.png', caption="SHAP Feature Importance", use_container_width=True)
        else:
            st.info("Chưa có shap_summary.png")
    
    with col_img2:
        if os.path.exists('shap_waterfall_example.png'):
            st.image('shap_waterfall_example.png', caption="SHAP Waterfall Example", use_container_width=True)
        else:
            st.info("Chưa có shap_waterfall_example.png")

# ============================================================
# TAB 5 — TRỢ LÝ AI
# ============================================================
with tab_ai:
    st.markdown('<div class="section-title">💬 Trợ lý AI — Phân tích thông minh</div>', unsafe_allow_html=True)

    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

    if not GROQ_API_KEY:
        st.warning("⚠️ Vui lòng cấu hình GROQ_API_KEY trong Streamlit Secrets để sử dụng tính năng này.")
    else:
        analysis_type = st.selectbox(
            "Chọn loại phân tích",
            ["Tổng quan doanh thu", "Phân tích sản phẩm", "Phân tích khách hàng", "Phân tích khu vực", "Đề xuất chiến lược"]
        )

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
<div class="report-footer">
    Báo cáo phân tích dữ liệu nội bộ &amp; Khuyến nghị sản phẩm &nbsp;|&nbsp; Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
""", unsafe_allow_html=True)
