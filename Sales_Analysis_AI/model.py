import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read CSV file
data = pd.read_csv("sales_data.csv")

# Input Features
data["Price_x_Qty"]=data["Price"]*data["Quantity"]
X = data[["Price", "Quantity","Price_x_Qty"]]

# Output Target
y = data["Total_Sales"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Display results
print("Actual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(predictions)

# Accuracy
accuracy = r2_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Save the trained model (so GUI can reuse it without retraining)
joblib.dump(model, "sales_model.pkl")
print("\nModel saved as sales_model.pkl")

# Sample prediction example
new_data = pd.DataFrame({"Price": [5000], "Quantity": [4], "Price_x_Qty":[5000*4]})
predicted = model.predict(new_data)
print(f"\nPredicted sales for Price=5000, Qty=4: ₹{predicted[0]:.2f}")
