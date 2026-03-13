import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rows=1000

cities=["Delhi","Mumbai","Bangalore","Hyderabad","Chennai"]
products=["Laptop","Mobile","Tablet","Headphones","Smartwatch"]
categories=["Electronics","Accessories","Gadgets"]

data= {
     "Order ID": np.arange(1, rows+1),
     "Order Date": pd.date_range(start="2023-01-01", periods=rows, freq='D'),
     "Customer ID": np.random.randint(1000, 2000, size=rows),
     "City": np.random.choice(cities, size=rows),
     "Product": np.random.choice(products, size=rows),
     "Category": np.random.choice(categories, size=rows),
     "Quantity": np.random.randint(1, 10, size=rows),
     "Price": np.random.randint(100, 5000, size=rows)
}

df=pd.DataFrame(data)

df["Revenue"]=df["Quantity"]*df["Price"]

df.loc[5, "City"] = np.nan
df.loc[10, "Price"] = np.nan
df.loc[20, "Quantity"] = np.nan

print(df)

df.to_csv("sales_data.csv", index=False)

df.info()

print(df.isnull().sum())             #checking missing values

df["City"].fillna("Unknown", inplace=True)
df["Price"].fillna(df["Price"].mean(), inplace=True)
df["Quantity"].fillna(df["Quantity"].median(), inplace=True)

print(df)

df.info()
df.describe()
df.head()

df.dropna(inplace=True)          #remove rows containing missing values

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Day"] = df["Order Date"].dt.day_name()

print(df.head())

Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)

IQR = Q3 - Q1
print(f"IQR: {IQR}")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

outliers = df[(df["Revenue"] < Q1 - 1.5*IQR) | (df["Revenue"] > Q3 + 1.5*IQR)]
print(f"Outliers:\n{outliers}")

df = df[(df["Revenue"] >= Q1 - 1.5*IQR) & (df["Revenue"] <= Q3 + 1.5*IQR)]
print(f"Data after removing outliers:\n{df}")

df.drop_duplicates(inplace=True)
print(f"Data after removing duplicates:\n{df}")

df.describe()
print(df.describe())

df["Revenue"].sum()                                #total business revenue
print(f"Total Revenue: {df['Revenue'].sum()}")

df["Revenue"].mean()

df["Order ID"].nunique()                    #number of unique orders

df.groupby("City")["Revenue"].sum()
print(df.groupby("City")["Revenue"].sum())

a=df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
print(f"Best Selling Product: {a.index[0]}") 

df.groupby("Category")["Revenue"].sum()
print(f"Revenue by Category:\n{df.groupby('Category')['Revenue'].sum()}")

df.groupby("Month")["Revenue"].sum()
print(f"Revenue by Month:\n{df.groupby('Month')['Revenue'].sum()}")

df.sort_values("Revenue", ascending=False).head(5)
print(f"Top 5 Orders by Revenue:\n{df.sort_values('Revenue', ascending=False).head(5)}")

df.sort_values("Revenue").head(5)
print(f"Lowest 5 Orders by Revenue:\n{df.sort_values('Revenue', ascending=True).head(5)}")

df["City"].value_counts()
print(f"No. of city orders :\n{df['City'].value_counts()}")

city_sales = df.groupby("City")["Revenue"].sum()

city_sales.plot(kind="bar")

plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Total Revenue")

plt.show()
plt.close()

monthly_sales = df.groupby("Month")["Revenue"].sum()
print(monthly_sales)
monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.show()
plt.close()

category_sales = df.groupby("Category")["Revenue"].sum()

category_sales.plot(kind="pie", autopct="%1.1f%%")

plt.title("Revenue by Category")
plt.ylabel("")
plt.show()
plt.close()

plt.figure()
data=df.groupby("City")["Revenue"].sum().reset_index()
data.plot(kind="bar", x="City", y="Revenue")
plt.hist(df["Revenue"], bins=20)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")

plt.show()
plt.close()

plt.figure()

plt.scatter(df["Quantity"], df["Revenue"])

plt.title("Quantity vs Revenue")
plt.xlabel("Quantity")
plt.ylabel("Revenue")

plt.show()
plt.close()

#pip install streamlit
#streamlit run sales_analytics_dashboard.py


'''Developed Retail Intelligence System using Python (Pandas, NumPy, Matplotlib, 
   Scikit-learn) to analyze 1000+ transactions, perform RFM segmentation, 
   detect revenue decline patterns, and forecast sales trends.'''






