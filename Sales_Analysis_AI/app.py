# import tkinter as tk
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# # Read dataset
# data = pd.read_csv("sales_data.csv")

# # Features and Target
# data["Price_x_Qty"]=data["Price"]*data["Quantity"]
# X = data[["Price", "Quantity","Price_x_Qty"]]
# y = data["Total_Sales"]

# # Train the model
# model = LinearRegression()
# model.fit(X, y)

# # Prediction function
# result_window = None # tracks the popup window

# def analysis_sales():
#     global result_window
#     try:
#         product_name = product_entry.get()
#         price = float(price_entry.get())
#         quantity = float(quantity_entry.get())

#         price_x_qty = price * quantity
#         prediction = model.predict(pd.DataFrame({
#             "Price": [price],
#             "Quantity": [quantity],
#             "Price_x_Qty": [price_x_qty]
#         }))

#         # Performance conclusion
#         if prediction[0] >= 60000:
#             performance = "High Sales 📈"
#             reason = "Predicted sales are well above average — strong demand expected."
#             perf_color = "#2E7D32"
#         elif prediction[0] >= 20000:
#             performance = "Moderate Sales ⚖️"
#             reason = "Predicted sales are close to the average — steady, typical demand."
#             perf_color = "#F9A825"
#         else:
#             performance = "Low Sales 📉"
#             reason = "Predicted sales are below average, likely due to low price or quantity."
#             perf_color = "#C62828"

#         # Close old popup if open
#         if result_window is not None and result_window.winfo_exists():
#             result_window.destroy()

#         # New popup window
#         result_window = tk.Toplevel(window)
#         result_window.title("Prediction Result")
#         result_window.geometry("480x600")
#         result_window.configure(bg="#f0f4f8")

#         tk.Label(result_window, text=f"Analysed Sales - {product_name}: ₹{prediction[0]:.2f}",
#                  font=("Arial", 14, "bold"), bg="#f0f4f8", fg="#2E7D32", wraplength=440).pack(pady=15)

#         tk.Label(result_window, text=f"Performance: {performance}\n{reason}",
#                  font=("Arial", 11, "bold"), bg="#f0f4f8", fg=perf_color,
#                  wraplength=420, justify="left").pack(pady=10)

#         fig, ax = plt.subplots(figsize=(7, 6))
#         ax.bar(["Price", "Analysed Sales"], [price, prediction[0]], color=["#4CAF50", "#FF9800"])
#         ax.set_xlabel("Parameters")
#         ax.set_ylabel("Amount (₹)")
#         ax.set_title(f"{product_name} - Input vs Analysed Sales (Qty: {int(quantity)})")

#         canvas = FigureCanvasTkAgg(fig, master=result_window)
#         canvas.draw()
#         canvas.get_tk_widget().pack(pady=10)

#     except ValueError:
#         result_label.config(text="Please enter valid numbers")

        
# # Create Window
# window = tk.Tk()
# window.title("E-Commerce Sales Analysis Using AI")
# window.geometry("1000x800")
# window.configure(bg="#f0f4f8")
# tk.Label(window,text="AI-Powered Sales Analysis",font=("Trebuchet MS",15,"bold"),fg="#1a5276",bg="#f0f4f8").pack(pady=10)
# chart_frame=tk.Frame(window,bg="#f0f4f8")
# chart_frame.pack(pady=10)

# tk.Label(window,text="Enter Product name",font=("Segoe UI",12),bg="#f0f4f8").pack(pady=5)
# product_entry=tk.Entry(window)
# product_entry.pack()

# # Price
# tk.Label(window, text="Enter Product Price",font=("Segoe UI",12),bg="#f0f4f8").pack(pady=5)
# price_entry = tk.Entry(window)
# price_entry.pack()

# # Quantity
# tk.Label(window, text="Enter Quantity",font=("Segoe UI",12),bg="#f0f4f8").pack(pady=5)
# quantity_entry = tk.Entry(window)
# quantity_entry.pack()

# # Predict Button
# tk.Button(window, text="Analysis Sales", command=analysis_sales,bg="#00AA41",fg="White",font=("Arial",10,"bold")).pack(pady=15)

# # Result
# result_label = tk.Label(window, text="", font=("Calibri", 14,"bold"),bg="#f0f4f8",fg="#2E7D32")
# result_label.pack(pady=10)
# performance_label = tk.Label(window, text="", font=("Arial", 12, "bold"), bg="#f0f4f8",wraplength=380,justify="left")
# performance_label.pack(pady=15)


# # Run Application
# window.mainloop()
import tkinter as tk
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================================================
# 1. READ DATASET
# ==========================================================

data = pd.read_csv("sales_data.csv")


# ==========================================================
# 2. DATA ANALYSIS
# ==========================================================

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


# Category-wise Sales
category_sales = (
    data.groupby("Category")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nCategory-wise Sales:")
print(category_sales)


# Create Category Sales Chart
category_sales.plot(kind="bar")

plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig("Category_Sales_Chart.png")
plt.show()


# ==========================================================
# 3. MACHINE LEARNING MODEL
# ==========================================================

# Create new feature
data["Price_x_Qty"] = data["Price"] * data["Quantity"]


# Input Features
X = data[
    ["Price", "Quantity", "Price_x_Qty"]
]


# Target
y = data["Total_Sales"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Test prediction
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = r2_score(
    y_test,
    predictions
)


print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# Save trained model
joblib.dump(
    model,
    "sales_model.pkl"
)

print("\nModel saved as sales_model.pkl")


# ==========================================================
# 4. GUI APPLICATION
# ==========================================================

result_window = None


def analysis_sales():

    global result_window

    try:

        # Get user input
        product_name = product_entry.get()

        price = float(
            price_entry.get()
        )

        quantity = float(
            quantity_entry.get()
        )


        # Calculate Price × Quantity
        price_x_qty = price * quantity


        # Create input DataFrame
        new_data = pd.DataFrame({

            "Price": [price],

            "Quantity": [quantity],

            "Price_x_Qty": [price_x_qty]

        })


        # Predict sales
        prediction = model.predict(
            new_data
        )

        predicted_sales = prediction[0]


        # ==================================================
        # PERFORMANCE ANALYSIS
        # ==================================================

        if predicted_sales >= 60000:

            performance = "High Sales 📈"

            reason = (
                "Predicted sales are well above average — "
                "strong demand expected."
            )

            perf_color = "#2E7D32"


        elif predicted_sales >= 20000:

            performance = "Moderate Sales ⚖️"

            reason = (
                "Predicted sales are close to the average — "
                "steady and typical demand."
            )

            perf_color = "#F9A825"


        else:

            performance = "Low Sales 📉"

            reason = (
                "Predicted sales are below average, "
                "likely due to low price or quantity."
            )

            perf_color = "#C62828"


        # ==================================================
        # CLOSE OLD RESULT WINDOW
        # ==================================================

        if (
            result_window is not None
            and result_window.winfo_exists()
        ):

            result_window.destroy()


        # ==================================================
        # RESULT WINDOW
        # ==================================================

        result_window = tk.Toplevel(
            window
        )

        result_window.title(
            "Prediction Result"
        )

        result_window.geometry(
            "600x650"
        )

        result_window.configure(
            bg="#f0f4f8"
        )


        # Product name
        tk.Label(

            result_window,

            text=f"Analysed Sales - {product_name}",

            font=("Arial", 16, "bold"),

            bg="#f0f4f8",

            fg="#1a5276"

        ).pack(pady=10)


        # Predicted sales
        tk.Label(

            result_window,

            text=f"Predicted Sales: ₹{predicted_sales:.2f}",

            font=("Arial", 14, "bold"),

            bg="#f0f4f8",

            fg="#2E7D32"

        ).pack(pady=5)


        # Performance
        tk.Label(

            result_window,

            text=f"Performance: {performance}",

            font=("Arial", 12, "bold"),

            bg="#f0f4f8",

            fg=perf_color

        ).pack(pady=5)


        # Reason
        tk.Label(

            result_window,

            text=reason,

            font=("Arial", 11),

            bg="#f0f4f8",

            fg=perf_color,

            wraplength=500,

            justify="left"

        ).pack(pady=5)


        # ==================================================
        # CREATE GRAPH
        # ==================================================

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        ax.bar(

            ["Price", "Predicted Sales"],

            [price, predicted_sales]

        )


        ax.set_xlabel(
            "Parameters"
        )

        ax.set_ylabel(
            "Amount (₹)"
        )

        ax.set_title(

            f"{product_name} - "
            f"Price vs Predicted Sales"

        )


        # Display graph inside Tkinter
        canvas = FigureCanvasTkAgg(

            fig,

            master=result_window

        )

        canvas.draw()


        canvas.get_tk_widget().pack(
            pady=15
        )


        plt.close(fig)


    except ValueError:

        error_label.config(

            text="Please enter valid numbers."

        )


# ==========================================================
# 5. MAIN WINDOW
# ==========================================================

window = tk.Tk()


window.title(
    "E-Commerce Sales Analysis Using AI"
)


window.geometry(
    "1000x800"
)


window.configure(
    bg="#f0f4f8"
)


# Main heading
tk.Label(

    window,

    text="AI-Powered Sales Analysis",

    font=("Trebuchet MS", 20, "bold"),

    fg="#1a5276",

    bg="#f0f4f8"

).pack(pady=20)


# Product Name
tk.Label(

    window,

    text="Enter Product Name",

    font=("Segoe UI", 12),

    bg="#f0f4f8"

).pack(pady=5)


product_entry = tk.Entry(

    window,

    width=30

)

product_entry.pack()


# Product Price
tk.Label(

    window,

    text="Enter Product Price",

    font=("Segoe UI", 12),

    bg="#f0f4f8"

).pack(pady=5)


price_entry = tk.Entry(

    window,

    width=30

)

price_entry.pack()


# Quantity
tk.Label(

    window,

    text="Enter Quantity",

    font=("Segoe UI", 12),

    bg="#f0f4f8"

).pack(pady=5)


quantity_entry = tk.Entry(

    window,

    width=30

)

quantity_entry.pack()


# Analysis Button
tk.Button(

    window,

    text="Analyse Sales",

    command=analysis_sales,

    bg="#00AA41",

    fg="white",

    font=("Arial", 11, "bold"),

    padx=20,

    pady=8

).pack(pady=20)


# Error message
error_label = tk.Label(

    window,

    text="",

    font=("Calibri", 12, "bold"),

    bg="#f0f4f8",

    fg="#C62828"

)

error_label.pack(pady=10)


# ==========================================================
# 6. RUN APPLICATION
# ==========================================================

window.mainloop()
