from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

import streamlit as st
from databricks import sql
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Load credentials
access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")
server_token = os.getenv("SERVER_HOSTNAME_TOKEN")
http_path = os.getenv("HTTP_PATH_TOKEN")

SERVER_HOSTNAME = server_token
HTTP_PATH = http_path
ACCESS_TOKEN = access_token


@st.cache_resource
def get_connection():
    """Create and cache database connection"""
    return sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=HTTP_PATH,
        access_token=ACCESS_TOKEN,
    )


def run_query(query):
    """Execute a SQL query and return as DataFrame"""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows, columns=columns)


# Page config
st.title("Business Metrics Dashboard")
st.markdown("Real-time insights from your gold layer data warehouse")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
    index=1
)

# Calculate date filter
if date_range == "Last 7 Days":
    days_back = 7
elif date_range == "Last 30 Days":
    days_back = 30
elif date_range == "Last 90 Days":
    days_back = 90
else:
    days_back = None

date_filter = f"AND t.event_date >= DATE_SUB(CURRENT_DATE(), {days_back})" if days_back else ""

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_resource.clear()
    st.rerun()

st.divider()

# ==================== SECTION 1: REVENUE METRICS ====================
st.header("Revenue Metrics")

with st.spinner("Loading revenue data..."):
    # Total revenue by status
    revenue_query = f"""
    SELECT 
        status,
        COUNT(DISTINCT transaction_id) as transaction_count,
        SUM(total) as total_revenue,
        AVG(total) as avg_order_value,
        SUM(quantity) as total_items
    FROM delta.`/Volumes/project_2/datalake/gold/fact_transactions/` t
    WHERE transaction_type = 'purchase'
        {date_filter}
    GROUP BY status
    ORDER BY total_revenue DESC
    """
    
    revenue_df = run_query(revenue_query)
    
    # Top-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    completed_rev = revenue_df[revenue_df['status'] == 'completed']['total_revenue'].sum()
    total_transactions = revenue_df['transaction_count'].sum()
    avg_order = revenue_df[revenue_df['status'] == 'completed']['avg_order_value'].mean()
    total_items = revenue_df['total_items'].sum()
    
    with col1:
        st.metric("Total Revenue", f"${completed_rev:,.2f}", 
                  help="Completed purchases only")
    
    with col2:
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col3:
        st.metric("Avg Order Value", f"${avg_order:,.2f}")
    
    with col4:
        st.metric("Items Sold", f"{total_items:,}")

# Revenue by status breakdown
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Status")
    fig_status = px.bar(
        revenue_df,
        x='status',
        y='total_revenue',
        color='status',
        text='total_revenue',
        color_discrete_map={
            'completed': '#28a745',
            'pending': '#ffc107', 
            'failed': '#dc3545',
            'cancelled': '#6c757d'
        }
    )
    fig_status.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_status.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue ($)")
    st.plotly_chart(fig_status, use_container_width=True)

with col2:
    st.subheader("Transaction Mix")
    fig_pie = px.pie(
        revenue_df,
        values='transaction_count',
        names='status',
        color='status',
        color_discrete_map={
            'completed': '#28a745',
            'pending': '#ffc107',
            'failed': '#dc3545',
            'cancelled': '#6c757d'
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Revenue trend over time
with st.spinner("Loading revenue trend..."):
    trend_query = f"""
    SELECT 
        event_date,
        SUM(CASE WHEN status = 'completed' THEN total ELSE 0 END) as revenue,
        COUNT(DISTINCT transaction_id) as transactions
    FROM delta.`/Volumes/project_2/datalake/gold/fact_transactions/` t
    WHERE transaction_type = 'purchase'
        {date_filter}
    GROUP BY event_date
    ORDER BY event_date
    """
    
    trend_df = run_query(trend_query)
    
    st.subheader("Revenue Trend")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df['event_date'],
        y=trend_df['revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#28a745', width=3),
        fill='tozeroy'
    ))
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ==================== SECTION 2: ACTIVE USERS ====================
st.header("Active Users")

with st.spinner("Loading user data..."):
    users_query = f"""
    SELECT 
        COUNT(DISTINCT t.user_id) as total_users,
        COUNT(DISTINCT CASE WHEN c.account_type = 'premium' THEN t.user_id END) as premium_users,
        COUNT(DISTINCT CASE WHEN c.account_type = 'standard' THEN t.user_id END) as standard_users,
        COUNT(DISTINCT CASE WHEN c.account_type = 'enterprise' THEN t.user_id END) as enterprise_users
    FROM delta.`/Volumes/project_2/datalake/gold/fact_transactions/` t
    LEFT JOIN delta.`/Volumes/project_2/datalake/gold/dim_customer/` c ON t.user_id = c.user_id
    WHERE t.transaction_type = 'purchase'
        {date_filter}
    """
    
    users_df = run_query(users_query)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Active Users", f"{users_df['total_users'][0]:,}")
    
    with col2:
        st.metric("Enterprise Users", f"{users_df['enterprise_users'][0]:,}",
                   delta=f"{users_df['enterprise_users'][0] / users_df['total_users'][0] * 100:.1f}%")

    with col3:
        st.metric("Premium Users", f"{users_df['premium_users'][0]:,}",
                  delta=f"{users_df['premium_users'][0] / users_df['total_users'][0] * 100:.1f}%")
    
    with col4:
        st.metric("Standard Users", f"{users_df['standard_users'][0]:,}",
                  delta=f"{users_df['standard_users'][0] / users_df['total_users'][0] * 100:.1f}%")

# User activity breakdown
with st.spinner("Loading user activity..."):
    activity_query = f"""
    SELECT 
        event_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT user_id) as unique_users
    FROM delta.`/Volumes/project_2/datalake/gold/fact_user_activity/`
    WHERE 1=1
        {date_filter.replace('t.', '')}
    GROUP BY event_type
    ORDER BY event_count DESC
    """
    
    activity_df = run_query(activity_query)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Activity by Type")
        fig_activity = px.bar(
            activity_df,
            x='event_type',
            y='event_count',
            color='event_type',
            text='event_count'
        )
        fig_activity.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_activity.update_layout(showlegend=False, xaxis_title="", yaxis_title="Event Count")
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        st.subheader("Unique Users by Event")
        fig_users = px.bar(
            activity_df,
            x='event_type',
            y='unique_users',
            color='event_type',
            text='unique_users'
        )
        fig_users.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_users.update_layout(showlegend=False, xaxis_title="", yaxis_title="Unique Users")
        st.plotly_chart(fig_users, use_container_width=True)

st.divider()

# ==================== SECTION 3: CATEGORY MIX ====================
st.header("Product & Category Performance")

with st.spinner("Loading category data..."):
    category_query = f"""
    SELECT 
        p.category,
        COUNT(DISTINCT t.transaction_id) as orders,
        SUM(t.total) as revenue,
        SUM(t.quantity) as units_sold,
        AVG(t.total) as avg_order_value
    FROM delta.`/Volumes/project_2/datalake/gold/fact_transactions/` t
    INNER JOIN delta.`/Volumes/project_2/datalake/gold/dim_product/` p 
        ON t.product_id = p.product_id
    WHERE t.transaction_type = 'purchase' 
        AND t.status = 'completed'
        {date_filter}
    GROUP BY p.category
    ORDER BY revenue DESC
    """
    
    category_df = run_query(category_query)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        fig_cat_rev = px.bar(
            category_df,
            x='category',
            y='revenue',
            color='category',
            text='revenue'
        )
        fig_cat_rev.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_cat_rev.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_cat_rev, use_container_width=True)
    
    with col2:
        st.subheader("Category Mix (Units)")
        fig_cat_units = px.pie(
            category_df,
            values='units_sold',
            names='category',
            hole=0.4
        )
        st.plotly_chart(fig_cat_units, use_container_width=True)

# Top products
with st.spinner("Loading top products..."):
    products_query = f"""
    SELECT 
        p.product_name,
        p.category,
        COUNT(DISTINCT t.transaction_id) as orders,
        SUM(t.total) as revenue,
        SUM(t.quantity) as units_sold
    FROM delta.`/Volumes/project_2/datalake/gold/fact_transactions/` t
    INNER JOIN delta.`/Volumes/project_2/datalake/gold/dim_product/` p 
        ON t.product_id = p.product_id
    WHERE t.transaction_type = 'purchase' 
        AND t.status = 'completed'
        {date_filter}
    GROUP BY p.product_name, p.category
    ORDER BY revenue DESC
    LIMIT 10
    """
    
    products_df = run_query(products_query)
    
    st.subheader("Top 10 Products by Revenue")
    
    # Format for display
    display_df = products_df.copy()
    display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.2f}")
    display_df['units_sold'] = display_df['units_sold'].apply(lambda x: f"{x:,}")
    display_df.columns = ['Product', 'Category', 'Orders', 'Revenue', 'Units Sold']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# Footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Showing data for: {date_range}")
