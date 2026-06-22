import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import pandas as pd
from datetime import datetime
import numpy as np
import os

# ===================== CẤU HÌNH TRANG =====================
st.set_page_config(
    page_title="Báo Cáo Dữ Liệu Nội Bộ",
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
    .report-header h1 { color: #ffffff; font-size: 18px; font-weight: 600; margin: 0; }
    .report-header .subtitle { color: #94a3b8; font-size: 12px; }
    .report-header .badge {
        background: rgba(59,130,246,0.2);
        color: #60a5fa;
        padding: 3px 12px;
        border-radius: 16px;
        font-size: 11px;
        border: 1px solid rgba(59,130,246,0.3);
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

    .report-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 10px;
        padding: 0.8rem 0;
        border-top: 1px solid #e5e9f0;
        margin-top: 1rem;
    }

    @media (max-width: 768px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 480px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown(f"""
<div class="report-header">
    <div>
        <h1>📊 Báo Cáo Dữ Liệu Nội Bộ</h1>
        <div class="subtitle">Hệ thống phân tích doanh thu & khuyến nghị sản phẩm</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span class="badge">Mô hình SVD</span>
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
tab_overview, tab_customer, tab_model, tab_ai = st.tabs([
    "📈 Tổng quan",
    "👥 Khách hàng",
    "🤖 Hiệu suất mô hình",
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
# TAB 3 — HIỆU SUẤT MÔ HÌNH SVD
# ============================================================
with tab_model:
    st.markdown('<div class="section-title">Hiệu suất mô hình gợi ý sản phẩm (SVD)</div>', unsafe_allow_html=True)

    eval_log = pd.DataFrame({
        'K': [1, 3, 5, 7],
        'Precision@K': [0.1847, 0.1914, 0.1728, 0.1401],
        'Recall@K':    [0.1847, 0.5742, 0.8638, 0.9808]
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

    st.info("""
    **Giải thích chỉ số:**
    - **Precision@K**: Trong K sản phẩm được gợi ý, tỷ lệ sản phẩm khách hàng thực sự mua.
    - **Recall@K**: Trong tổng số sản phẩm khách hàng đã mua, tỷ lệ được mô hình tìm đúng trong top K gợi ý.
    - Precision cao nhất ở K=3, Recall tăng dần theo K — phù hợp với gợi ý top 3–5 sản phẩm.
    """)

# ============================================================
# TAB 4 — TRỢ LÝ AI
# ============================================================
with tab_ai:
    st.markdown('<div class="section-title">Trợ lý AI — Phân tích thông minh</div>', unsafe_allow_html=True)

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
                        model="llama3-70b-8192",
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
    Báo cáo phân tích dữ liệu nội bộ &nbsp;|&nbsp; Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
""", unsafe_allow_html=True)
