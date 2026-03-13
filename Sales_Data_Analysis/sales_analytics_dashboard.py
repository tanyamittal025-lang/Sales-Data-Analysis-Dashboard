import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import calendar
import openpyxl
import io


st.title("Sales Data Analysis Dashboard")

# load data
df = pd.read_csv("sales_data.csv")

st.subheader("Dataset Preview")
st.write(df.head())

#This lets users choose a date range to analyse sales
df["Order Date"] = pd.to_datetime(df["Order Date"])

start_date = st.date_input("Start Date", df["Order Date"].min())
end_date = st.date_input("End Date", df["Order Date"].max())

filtered_df = df[(df["Order Date"] >= pd.to_datetime(start_date)) & 
                 (df["Order Date"] <= pd.to_datetime(end_date))]

# Revenue by City in date range
st.subheader("Revenue by City")
city_sales = filtered_df.groupby("City")["Revenue"].sum()

fig, ax = plt.subplots()
city_sales.plot(kind="bar", ax=ax)
st.pyplot(fig)
plt.savefig("revenue_chart.png")


# Revenue by City
st.subheader("Revenue by City")
city_sales = df.groupby("City")["Revenue"].sum()

fig, ax = plt.subplots()
city_sales.plot(kind="bar", ax=ax)

st.pyplot(fig)

# Monthly Revenue
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.month

st.subheader("Monthly Revenue Trend")

monthly_sales = df.groupby("Month")["Revenue"].sum().reset_index()

fig2, ax2 = plt.subplots()
monthly_sales.plot(kind="line", marker="o", ax=ax2)

st.pyplot(fig2)

#Revenue by Product Category
st.subheader("Revenue by Category")
category_sales = df.groupby("Category")["Revenue"].sum()
fig3, ax3 = plt.subplots()

category_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax3)
ax3.set_ylabel("")

st.pyplot(fig3)

#Products which generate the most revenue.
st.subheader("Top 5 Products by Revenue")

top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(5)

fig4, ax4 = plt.subplots()
top_products.plot(kind="bar", ax=ax4)

st.pyplot(fig4)



# RFM Analysis

rfm = df.groupby("Customer ID").agg({
    "Order Date": lambda x: (df["Order Date"].max() - x.max()).days,
    "Order ID": "count",
    "Revenue": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["R_score"] = rfm["Recency"].rank(ascending=False)
rfm["F_score"] = rfm["Frequency"].rank()
rfm["M_score"] = rfm["Monetary"].rank()

def segment(row):
    
    if row["Monetary"] > rfm["Monetary"].quantile(0.75):
        return "VIP"
    elif row["Recency"] > rfm["Recency"].quantile(0.75):
        return "At Risk"
    else:
        return "Regular"
rfm["Segment"] = rfm.apply(segment, axis=1)

st.subheader("Customer RFM Analysis")
st.write(rfm.head())
print(rfm.head())

segment_count = rfm["Segment"].value_counts()

st.subheader("Customer Segment Distribution")
st.bar_chart(segment_count)

plt.savefig("sales_trend.png")

# displaying the 3-month moving average of monthly sales
monthly_sales = df.groupby("Month")["Revenue"].sum().reset_index()

monthly_sales["Moving_Avg"] = monthly_sales["Revenue"].rolling(3).mean()

st.subheader("3 Month Moving Average")

st.line_chart(monthly_sales.set_index("Month")[["Revenue","Moving_Avg"]])

#decline in sales month over month

monthly_sales["Change"] = monthly_sales["Revenue"].diff()

def colours(val):
    color = "red" if val < 0 else "green"
    return f"color: {color}"

st.subheader("Month-to-Month Change")
st.write(monthly_sales.style.applymap(colours, subset=["Change"]))

#Forecast Next Month Revenue(Linear Regression)
# convert month to number
monthly_sales["Month_num"] = range(1, len(monthly_sales)+1)

X = monthly_sales[["Month_num"]]
y = monthly_sales["Revenue"]

model = LinearRegression()
model.fit(X, y)

next_month = np.array([[len(monthly_sales)+1]])
forecast = model.predict(next_month)

st.subheader("Next Month Revenue Forecast")
st.write("Predicted Revenue:", round(forecast[0],2))

#Automated Insight Generator

def generate_insights(df):

    insights = []

    # highest revenue month
    monthly = df.groupby("Month")["Revenue"].sum()
    peak_month = monthly.idxmax()
    peak_month_name = calendar.month_name[peak_month]
    insights.append(f"Revenue peaked in {peak_month_name}")

    # city decline
    city_sales = df.groupby("City")["Revenue"].sum()
    lowest_city = city_sales.idxmin()
    insights.append(f"{lowest_city} shows lowest revenue performance")

    # top customers contribution
    customer_sales = df.groupby("Customer ID")["Revenue"].sum()
    top10 = customer_sales.quantile(0.9)
    top_customers = customer_sales[customer_sales >= top10].sum()

    percent = round((top_customers/customer_sales.sum())*100,2)

    insights.append(f"Top 10% customers contribute {percent}% of total revenue")

    return insights

st.subheader("Automated Business Insights")

insights = generate_insights(df)

for i in insights:
    st.write("•", i)

st.download_button(
    label="Download Filtered Data",
    data=df.to_csv(index=False),
    file_name="sales_report.csv",
    mime="text/csv"
)


output = io.BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    monthly_sales.to_excel(writer, index=False, sheet_name="Monthly Report")

st.download_button(
    "Download Excel Report",
    data=output.getvalue(),
    file_name="sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

df.to_csv("cleaned_sales_data.csv", index=False)

df.to_excel("sales_report.xlsx", index=False)

