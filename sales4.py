import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("retail_sales_dataset.csv")

df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['Date'])

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Month_Period'] = df['Date'].dt.to_period('M')

# --------------------------------------------------
# HEADER WITH LOGO
# --------------------------------------------------
logo = Image.open("download (1).jpg")

col1, col2 = st.columns([1, 6])

with col1:
    st.image(logo, width=120)

with col2:
    st.markdown(
        "<h1 style='margin-top:25px;'>Sales Prediction Dashboard</h1>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Controls 📊")

customer_search = st.sidebar.text_input("Search Customer ID")

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=df['Product Category'].unique(),
    default=df['Product Category'].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_filter = st.sidebar.slider(
    "Age Range",
    age_min,
    age_max,
    (age_min, age_max)
)

month_filter = st.sidebar.multiselect(
    "Month",
    options=df['Month_Name'].unique(),
    default=df['Month_Name'].unique()
)

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------
filtered_df = df.copy()

if customer_search:
    filtered_df = filtered_df[
        filtered_df['Customer ID']
        .astype(str)
        .str.contains(customer_search, case=False)
    ]

filtered_df = filtered_df[
    (filtered_df['Product Category'].isin(category_filter)) &
    (filtered_df['Gender'].isin(gender_filter)) &
    (filtered_df['Age'].between(age_filter[0], age_filter[1])) &
    (filtered_df['Month_Name'].isin(month_filter))
]

# --------------------------------------------------
# MONTHLY SALES AGGREGATION
# --------------------------------------------------
monthly_sales = (
    filtered_df
    .groupby('Month_Period')['Total Amount']
    .sum()
    .reset_index()
)

# Convert Period → Timestamp → Month Name
monthly_sales['Month_Period'] = monthly_sales['Month_Period'].dt.to_timestamp()
monthly_sales['Month_Name'] = monthly_sales['Month_Period'].dt.strftime('%B')

# --------------------------------------------------
# FIX MONTH SORTING (IMPORTANT)
# --------------------------------------------------
monthly_sales['Month_Number'] = monthly_sales['Month_Period'].dt.month
monthly_sales = monthly_sales.sort_values(by='Month_Number')

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
total_revenue = monthly_sales['Total Amount'].sum()
avg_monthly_revenue = monthly_sales['Total Amount'].mean()

# --------------------------------------------------
# LINEAR REGRESSION (JAN–DEC → NEXT JAN)
# --------------------------------------------------
predicted_next_jan = 0
expected_growth = 0
last_month_revenue = 0

if len(monthly_sales) >= 12:
    last_12 = monthly_sales.tail(12).copy()

    X = np.arange(len(last_12))
    y = last_12['Total Amount'].values

    model = np.polyfit(X, y, 1)

    predicted_next_jan = model[0] * 12 + model[1]
    last_month_revenue = y[-1]

    if last_month_revenue > 0:
        expected_growth = (
            (predicted_next_jan - last_month_revenue)
            / last_month_revenue
        ) * 100

    last_12['Forecast'] = model[0] * X + model[1]

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
st.markdown("## 📌 Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
k2.metric("Avg Monthly Revenue", f"₹{avg_monthly_revenue:,.0f}")
k3.metric("Predicted Next Month Revenue", f"₹{predicted_next_jan:,.0f}")
k4.metric(
    "Expected Growth",
    f"{expected_growth:.2f}%",
    delta=f"₹{predicted_next_jan - last_month_revenue:,.0f}"
)

# --------------------------------------------------
# MONTHLY SALES TREND
# --------------------------------------------------
st.markdown("## 📈 Monthly Revenue Trend")

st.line_chart(
    monthly_sales.set_index('Month_Name')['Total Amount']
)

# --------------------------------------------------
# FORECAST VS ACTUAL
# --------------------------------------------------
if len(monthly_sales) >= 12:
    st.markdown("## 🔮 Forecast vs Actual")

    st.line_chart(
        last_12.set_index('Month_Name')[['Total Amount', 'Forecast']]
    )

# --------------------------------------------------
# KEY INSIGHTS
# --------------------------------------------------

# ADDITIONAL VISUALIZATIONS
# --------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Additional Insights")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🏷️ Revenue by Product Category")
    
    # Calculate revenue by category
    category_revenue = (
        filtered_df.groupby('Product Category')['Total Amount']
        .sum()
        .reset_index()
        .sort_values('Total Amount', ascending=False)
    )
    
    if not category_revenue.empty:
        fig3 = px.bar(
            category_revenue,
            x='Product Category',
            y='Total Amount',
            color='Product Category',
            text_auto='.2s'
        )
        fig3.update_layout(
            xaxis_title="Product Category",
            yaxis_title="Revenue (₹)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data available for selected filters")

with col4:
    st.markdown("### 👥 Revenue by Gender")
    
    # Calculate revenue by gender
    gender_revenue = (
        filtered_df.groupby('Gender')['Total Amount']
        .sum()
        .reset_index()
    )
    
    if not gender_revenue.empty:
        fig4 = px.pie(
            gender_revenue,
            values='Total Amount',
            names='Gender',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        fig4.update_layout(
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data available for selected filters")







st.markdown("""
### 📊 Key Sales & Revenue Insights

<ul style="line-height:1.6;">
  <li><b>Total Financial Impact:</b> The dataset records a total revenue of <b>$455,730.00</b> across <b>1,000 transactions</b>, with an Average Order Value (AOV) of <b>$456.64</b>.</li>

  <li><b>Leading Category:</b> <b>Electronics</b> is the top revenue generator ($156,635), though <b>Clothing</b> follows extremely closely ($155,580), indicating a balanced demand for lifestyle and tech products.</li>

  <li><b>Peak Sales Period:</b> <b>Quarter 2 (Q2)</b> emerged as the strongest period for sales ($128,975), with <b>May</b> and <b>February</b> being the highest-performing months individually.</li>

  <li><b>Demographic Dominance:</b>
    <ul>
      <li><b>Gender:</b> Female customers contribute slightly more to total revenue ($232,690) compared to Male customers ($223,040).</li>
      <li><b>Age Profile:</b> The <b>46–60 age group</b> is the most valuable segment ($147,755), followed closely by the <b>31–45 group</b>.</li>
    </ul>
  </li>

  <li><b>Quantity Trends:</b> A total of <b>2,507 items</b> were sold, indicating customers typically purchase 2–3 items per transaction.</li>
</ul>
""", unsafe_allow_html=True)




# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
with st.expander("📋 View Filtered Data Preview"):
    st.dataframe(
        filtered_df.sort_values('Date', ascending=False).head(100),
        use_container_width=True
    )
    
    # Download button for filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )