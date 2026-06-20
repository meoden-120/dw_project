import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Data Warehouse Gold Dashboard", 
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { font-size: 18px; }
    h1 { font-size: 24px !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 18px !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 16px !important; }
    p, li, div { font-size: 16px !important; }
    
    .header-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 0.6rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-title h1 { font-size: 24px !important; margin: 0; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.3rem;
    }
    .metric-value { font-size: 1.4rem; font-weight: bold; line-height: 1.2; }
    .metric-label { font-size: 0.75rem; opacity: 0.9; margin-top: 2px; }
    .metric-growth { font-size: 0.7rem; margin-top: 2px; }
    .growth-positive { color: #4ade80; }
    .growth-negative { color: #f87171; }
    
    .stTabs [data-baseweb="tab-list"] { font-size: 16px; gap: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; }
    .dataframe { font-size: 15px !important; }
    .stButton button { font-size: 16px !important; padding: 0.3rem 1rem !important; }
    .stSelectbox, .stTextInput { font-size: 16px !important; }
    hr { margin: 0.8rem 0 !important; }
    .js-plotly-plot { height: 320px !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header-title"><h1>Data Warehouse Gold Dashboard</h1></div>', unsafe_allow_html=True)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_data
def load_base_data():
    conn = duckdb.connect()
    query = """
        SELECT 
            "product_name",
            "category",
            "customer_name",
            "customer_id",
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
            "gender",
            "city",
            "state",
            "order_date"
        FROM 'NKDL_Project.csv'
    """
    df = conn.execute(query).df()
    conn.close()
    
    # Thêm các cột tính toán
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    return df

df_raw = load_base_data()

# Sidebar
st.sidebar.markdown("### Bo Loc")

categories = ["Tat ca"] + list(df_raw["category"].dropna().unique())
selected_category = st.sidebar.selectbox("Danh muc", categories)

years = ["Tat ca"] + list(df_raw["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Nam", years)

months = ["Tat ca"] + list(df_raw["month"].dropna().unique())
selected_month = st.sidebar.selectbox("Thang", months)

genders = ["Tat ca"] + list(df_raw["gender"].dropna().unique())
selected_gender = st.sidebar.selectbox("Gioi tinh", genders)

# Filter nâng cao - Khoảng giá
st.sidebar.markdown("### Khoang Gia")
min_price = float(df_raw['min_price'].min())
max_price = float(df_raw['max_price'].max())
price_range = st.sidebar.slider(
    "Khoang gia",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=1000.0
)

# Tìm kiếm sản phẩm
st.sidebar.markdown("### Tim Kiem")
search_product = st.sidebar.text_input("Ten san pham", placeholder="Nhap ten san pham...")

# Apply filters
df_filtered = df_raw.copy()
if selected_category != "Tat ca":
    df_filtered = df_filtered[df_filtered["category"] == selected_category]
if selected_year != "Tat ca":
    df_filtered = df_filtered[df_filtered["year"] == selected_year]
if selected_month != "Tat ca":
    df_filtered = df_filtered[df_filtered["month"] == selected_month]
if selected_gender != "Tat ca":
    df_filtered = df_filtered[df_filtered["gender"] == selected_gender]

# Filter theo khoảng giá
df_filtered = df_filtered[
    (df_filtered['min_price'] >= price_range[0]) & 
    (df_filtered['max_price'] <= price_range[1])
]

# Filter theo tìm kiếm
if search_product:
    df_filtered = df_filtered[df_filtered['product_name'].str.contains(search_product, case=False)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Tong Quan", "Phan Tich Nang Cao", "Khach Hang", "Tro Ly AI"])

with tab1:
    # Metrics
    total_rev = df_filtered["revenue"].sum()
    total_qty = df_filtered["quantity"].sum()
    total_ord = df_filtered["orders"].sum()
    avg_rev = df_filtered["avg_revenue"].mean()
    avg_order_value = total_rev / total_ord if total_ord > 0 else 0
    unique_customers = len(df_filtered['customer_id'].unique())
    
    # Tính MoM growth - SỬA LỖI
    mom_growth = 0
    unique_months = sorted(df_filtered['year_month'].unique())
    
    if len(unique_months) >= 2:
        current_month_str = unique_months[-1]
        prev_month_str = unique_months[-2]
        
        current_month_revenue = df_filtered[df_filtered['year_month'] == current_month_str]['revenue'].sum()
        prev_month_revenue = df_filtered[df_filtered['year_month'] == prev_month_str]['revenue'].sum()
        
        if prev_month_revenue > 0:
            mom_growth = ((current_month_revenue - prev_month_revenue) / prev_month_revenue) * 100
    
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_rev:,.0f}</div>
            <div class="metric-label">Doanh Thu</div>
            <div class="metric-growth {'growth-positive' if mom_growth > 0 else 'growth-negative'}">
                {'↑' if mom_growth > 0 else '↓'} {abs(mom_growth):.1f}% MoM
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_qty:,.0f}</div>
            <div class="metric-label">So Luong</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_ord:,.0f}</div>
            <div class="metric-label">Don Hang</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_rev:,.0f}</div>
            <div class="metric-label">Doanh Thu TB</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_order_value:,.0f}</div>
            <div class="metric-label">Gia Tri Don TB</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_customers:,}</div>
            <div class="metric-label">Khach Hang</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Biểu đồ
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Doanh Thu Theo Thoi Gian")
        df_line = df_filtered.groupby("year_month")["revenue"].sum().reset_index()
        df_line = df_line.sort_values("year_month")
        fig_line = px.line(df_line, x="year_month", y="revenue", 
                          markers=True, 
                          color_discrete_sequence=["#667eea"])
        fig_line.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=30, b=30),
            xaxis_title=None,
            yaxis_title="Doanh thu"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        st.subheader("Doanh Thu Theo Danh Muc")
        df_cat = df_filtered.groupby("category")["revenue"].sum().reset_index()
        df_cat = df_cat.sort_values("revenue", ascending=False)
        fig_cat = px.bar(df_cat, x="category", y="revenue",
                        color="revenue",
                        color_continuous_scale="Blues")
        fig_cat.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=30, b=30),
            xaxis_title=None,
            yaxis_title="Doanh thu"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Top 10 San Pham")
        df_bar = df_filtered.groupby("product_name")["revenue"].sum().reset_index()
        df_bar = df_bar.sort_values(by="revenue", ascending=True).tail(10)
        fig_bar = px.bar(df_bar, x="revenue", y="product_name", 
                        orientation="h", 
                        color="revenue",
                        color_continuous_scale="Viridis")
        fig_bar.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=30, b=30),
            xaxis_title="Doanh thu",
            yaxis_title=None
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Doanh Thu Theo Khu Vuc")
        df_state = df_filtered.groupby("state")["revenue"].sum().reset_index()
        df_state = df_state.sort_values("revenue", ascending=False).head(10)
        fig_state = px.bar(df_state, x="state", y="revenue",
                          color="revenue",
                          color_continuous_scale="Oranges")
        fig_state.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=30, b=30),
            xaxis_title=None,
            yaxis_title="Doanh thu"
        )
        st.plotly_chart(fig_state, use_container_width=True)
    
    st.divider()
    
    # Export data
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Tai du lieu CSV", use_container_width=True):
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(df_filtered)
            st.download_button(
                label="Tai xuong",
                data=csv,
                file_name=f'data_export_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    with st.expander("Xem du lieu chi tiet", expanded=False):
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=300,
            column_config={
                "revenue": st.column_config.NumberColumn("Doanh Thu", format="%d"),
                "quantity": st.column_config.NumberColumn("So Luong"),
                "orders": st.column_config.NumberColumn("Don Hang")
            }
        )

with tab2:
    st.subheader("Phan Tich Nang Cao")
    
    # Phân tích ABC
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Phan Tich ABC (Pareto)")
        df_abc = df_filtered.groupby("product_name")["revenue"].sum().reset_index()
        df_abc = df_abc.sort_values("revenue", ascending=False)
        df_abc["cumulative_pct"] = df_abc["revenue"].cumsum() / df_abc["revenue"].sum() * 100
        df_abc["category"] = "C"
        df_abc.loc[df_abc["cumulative_pct"] <= 80, "category"] = "A"
        df_abc.loc[(df_abc["cumulative_pct"] > 80) & (df_abc["cumulative_pct"] <= 95), "category"] = "B"
        
        abc_stats = df_abc.groupby("category").agg({
            "product_name": "count",
            "revenue": "sum"
        }).reset_index()
        abc_stats.columns = ["Category", "So SP", "Doanh Thu"]
        abc_stats["% Doanh Thu"] = (abc_stats["Doanh Thu"] / abc_stats["Doanh Thu"].sum() * 100).round(1)
        
        st.dataframe(abc_stats, use_container_width=True)
        
        fig_abc = px.bar(df_abc, x="product_name", y="revenue", 
                        color="category",
                        color_discrete_map={"A": "#4ade80", "B": "#fbbf24", "C": "#f87171"},
                        title="Phan loai san pham ABC")
        fig_abc.update_layout(height=300, xaxis_title=None)
        st.plotly_chart(fig_abc, use_container_width=True)
    
    with col2:
        st.subheader("Phan Tich RFM")
        today = pd.Timestamp.now()
        df_rfm = df_filtered.groupby("customer_id").agg({
            "order_date": lambda x: (today - pd.to_datetime(x).max()).days,
            "orders": "count",
            "revenue": "sum"
        }).reset_index()
        df_rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
        
        # Phân hạng
        df_rfm["r_score"] = pd.qcut(df_rfm["recency"], 4, labels=[4, 3, 2, 1])
        df_rfm["f_score"] = pd.qcut(df_rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
        df_rfm["m_score"] = pd.qcut(df_rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4])
        df_rfm["rfm_score"] = df_rfm["r_score"].astype(str) + df_rfm["f_score"].astype(str) + df_rfm["m_score"].astype(str)
        
        def classify_customer(row):
            if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
                return "VIP"
            elif row["r_score"] >= 3 and row["f_score"] >= 3:
                return "Than Thiet"
            elif row["r_score"] >= 2:
                return "Tiem Nang"
            else:
                return "Roi Bo"
        
        df_rfm["segment"] = df_rfm.apply(classify_customer, axis=1)
        
        segment_stats = df_rfm.groupby("segment").agg({
            "customer_id": "count",
            "monetary": "sum"
        }).reset_index()
        segment_stats.columns = ["Phan Khuc", "So KH", "Doanh Thu"]
        
        st.dataframe(segment_stats, use_container_width=True)
        
        fig_rfm = px.pie(segment_stats, values="Doanh Thu", names="Phan Khuc",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        title="Phan bo doanh thu theo phan khuc")
        fig_rfm.update_layout(height=300)
        st.plotly_chart(fig_rfm, use_container_width=True)
    
    st.divider()
    
    # Heatmap
    st.subheader("Heatmap Doanh Thu Theo Ngay Trong Tuan")
    
    df_filtered['day_of_week'] = df_filtered['order_date'].dt.day_name()
    df_filtered['hour'] = df_filtered['order_date'].dt.hour
    
    heatmap_data = df_filtered.pivot_table(
        values='revenue', 
        index='day_of_week', 
        columns='hour', 
        aggfunc='sum', 
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Doanh thu theo ngay trong tuan va gio"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Phan Tich Khach Hang")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        df_gender = df_filtered.groupby("gender").agg({
            "revenue": "sum",
            "quantity": "sum",
            "orders": "sum"
        }).reset_index()
        fig_gender = px.bar(df_gender, x="gender", y="revenue",
                           title="Doanh thu theo gioi tinh",
                           color="gender",
                           color_discrete_sequence=["#FF6B6B", "#4ECDC4"])
        fig_gender.update_layout(height=350)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        df_loyalty = df_filtered.groupby("customer_name")["loyalty_points"].sum().reset_index()
        df_loyalty = df_loyalty.sort_values("loyalty_points", ascending=True).tail(10)
        fig_loyalty = px.bar(df_loyalty, x="loyalty_points", y="customer_name",
                            orientation="h",
                            title="Top 10 khach hang diem tich luy",
                            color="loyalty_points",
                            color_continuous_scale="Oranges")
        fig_loyalty.update_layout(height=350)
        st.plotly_chart(fig_loyalty, use_container_width=True)
    
    st.divider()
    
    st.subheader("Khach Hang VIP")
    df_customer = df_filtered.groupby("customer_name").agg({
        "revenue": "sum",
        "orders": "sum",
        "loyalty_points": "sum"
    }).reset_index()
    df_customer["avg_order_value"] = df_customer["revenue"] / df_customer["orders"]
    df_customer = df_customer.sort_values("revenue", ascending=False).head(10)
    
    st.dataframe(
        df_customer,
        use_container_width=True,
        height=300,
        column_config={
            "customer_name": "Ten KH",
            "revenue": st.column_config.NumberColumn("Doanh Thu", format="%d"),
            "orders": "Don Hang",
            "loyalty_points": "Diem TL",
            "avg_order_value": st.column_config.NumberColumn("Gia Tri TB", format="%d")
        }
    )
    
    # Top khách hàng tiềm năng
    if 'df_rfm' in locals():
        st.subheader("Khach Hang Tiem Nang")
        potential_customers = df_rfm[df_rfm['segment'] == 'Tiem Nang'].head(10)
        if len(potential_customers) > 0:
            st.dataframe(
                potential_customers[['customer_id', 'recency', 'frequency', 'monetary']],
                use_container_width=True,
                height=200,
                column_config={
                    "customer_id": "Ma KH",
                    "recency": "Ngay gan day",
                    "frequency": "So don",
                    "monetary": "Doanh thu"
                }
            )
        else:
            st.info("Khong co khach hang tiem nang")

with tab4:
    st.subheader("Tro Ly AI")
    
    analysis_type = st.selectbox(
        "Chon phan tich",
        ["Tong quan", "San pham", "Khach hang", "Khu vuc", "De xuat chien luoc"]
    )
    
    if analysis_type == "Tong quan":
        df_ai_summary = df_filtered.groupby("product_name").agg({
            "revenue": "sum", 
            "quantity": "sum",
            "orders": "sum"
        }).reset_index()
        data_summary = df_ai_summary.sort_values(by="revenue", ascending=False).head(10).to_string(index=False)
        prompt_instruction = "Du lieu top 10 san pham ve doanh thu, so luong va don hang."
    elif analysis_type == "San pham":
        df_ai_summary = df_filtered.groupby(["category", "product_name"]).agg({
            "revenue": "sum", 
            "quantity": "sum",
            "avg_revenue": "mean"
        }).reset_index()
        data_summary = df_ai_summary.sort_values(by="revenue", ascending=False).head(10).to_string(index=False)
        prompt_instruction = "Du lieu top 10 san pham theo danh muc."
    elif analysis_type == "Khach hang":
        df_ai_summary = df_filtered.groupby(["customer_name", "gender"]).agg({
            "revenue": "sum",
            "orders": "sum",
            "loyalty_points": "sum"
        }).reset_index()
        data_summary = df_ai_summary.sort_values(by="revenue", ascending=False).head(10).to_string(index=False)
        prompt_instruction = "Du lieu top 10 khach hang."
    elif analysis_type == "Khu vuc":
        df_ai_summary = df_filtered.groupby(["state", "city"]).agg({
            "revenue": "sum",
            "quantity": "sum",
            "orders": "sum"
        }).reset_index()
        data_summary = df_ai_summary.sort_values(by="revenue", ascending=False).head(10).to_string(index=False)
        prompt_instruction = "Du lieu top 10 khu vuc."
    else:  # De xuat chien luoc
        df_ai_summary = df_filtered.groupby("product_name").agg({
            "revenue": "sum", 
            "quantity": "sum"
        }).reset_index()
        data_summary = df_ai_summary.sort_values(by="revenue", ascending=False).head(10).to_string(index=False)
        prompt_instruction = "Du lieu top 10 san pham ve doanh thu va so luong. Hay de xuat chien luoc kinh doanh."
    
    if st.button("Phan tich", use_container_width=True):
        with st.spinner("Dang phan tich..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                total_revenue = df_filtered["revenue"].sum()
                total_orders = df_filtered["orders"].sum()
                avg_loyalty = df_filtered["avg_loyalty_points"].mean()
                unique_customers = len(df_filtered['customer_id'].unique())
                
                prompt = f"""
                {prompt_instruction}
                
                Du lieu:
                {data_summary}
                
                Thong tin tong quan:
                - Tong doanh thu: {total_revenue:,.0f} VND
                - Tong don hang: {total_orders}
                - Khach hang: {unique_customers}
                - Diem TL TB: {avg_loyalty:.0f}
                
                Yeu cau:
                1. 3-5 nhan xet quan trong
                2. 3-5 de xuat chien luoc
                3. Co hoi va thach thuc
                
                Tra loi bang tieng Viet, markdown.
                """
                
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1024
                )
                
                st.success("Hoan tat!")
                st.markdown(chat_completion.choices[0].message.content)
                
            except Exception as e:
                st.error(f"Loi: {e}")

# Footer
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 0.5rem; font-size: 12px;">
    Data Warehouse Gold Dashboard | Cap nhat: {datetime.now().strftime("%d/%m/%Y %H:%M")}
</div>
""", unsafe_allow_html=True)