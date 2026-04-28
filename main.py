# -------------------------------
# House Price Prediction Project
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. CREATE SYNTHETIC DATASET
# -------------------------------

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "area": np.random.randint(500, 5000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "age": np.random.randint(0, 30, n),
    "parking": np.random.randint(0, 2, n),
    "location": np.random.choice(["urban", "suburban", "rural"], n)
})

# Generate price (target)
data["price"] = (
    data["area"] * 300 +
    data["bedrooms"] * 50000 +
    data["bathrooms"] * 30000 -
    data["age"] * 10000 +
    data["parking"] * 20000 +
    np.random.randint(-50000, 50000, n)
)

print("\nDataset Preview:\n", data.head())

# -------------------------------
# 2. DATA PREPROCESSING
# -------------------------------

# Encode location
data = pd.get_dummies(data, columns=["location"], drop_first=True)

# Features & target
X = data.drop("price", axis=1)
y = data["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. MODEL TRAINING
# -------------------------------

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------------------------------
# 4. EVALUATION
# -------------------------------

def evaluate(model, name):
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\n{name} Performance:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    
    return preds

lr_preds = evaluate(lr, "Linear Regression")
rf_preds = evaluate(rf, "Random Forest")

# -------------------------------
# 5. VISUALIZATION
# -------------------------------

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.show()

# Actual vs Predicted
plt.scatter(y_test, rf_preds)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("outputs/prediction.png")
plt.show()

# -------------------------------
# 6. PREDICTION FUNCTION
# -------------------------------

def predict_price(model):
    print("\nEnter House Details:")
    
    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    age = int(input("Age: "))
    parking = int(input("Parking (0/1): "))
    urban = int(input("Urban (1/0): "))
    suburban = int(input("Suburban (1/0): "))
    
    features = np.array([[area, bedrooms, bathrooms, age, parking, urban, suburban]])
    
    price = model.predict(features)
    print(f"\nPredicted Price: ₹{price[0]:,.2f}")

# Run prediction
predict_price(rf)