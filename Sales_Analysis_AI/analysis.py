import pandas as pd
import matplotlib.pyplot as plt
# Read CSV file
data = pd.read_csv("sales_data.csv")

print("E-Commerce Sales Data")
print(data)

print("\nTotal Number of Products:")
print(data["Product"].count())

print("\nTotal Sales:")
print(data["Total_Sales"].sum())

print("\nAverage Product Price:")
print(data["Price"].mean())

print("\nHighest Sales:")
print(data["Total_Sales"].max())

print("\nLowest Sales:")
print(data["Total_Sales"].min())

category_sales = data.groupby("Category")['Total_Sales'].sum().sort_values(ascending=False)
print("\nCategory-wise Sales:")
print(category_sales)

category_sales.plot(kind='bar',color='skyblue')
plt.title('Category-wise Sales')
plt.xlabel('Category')
plt.ylabel('Total_Sales')
plt.tight_layout()
plt.savefig('Category_Sales_Chart.png')
plt.show()