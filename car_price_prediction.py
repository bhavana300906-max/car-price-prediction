"""Used car selling price prediction using Linear Regression."""

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/car data.csv")
CURRENT_YEAR = date.today().year


def load_data(path=DATA_PATH):
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Place 'car data.csv' in the data/ folder.")
    df = pd.read_csv(path)
    print(df.head())
    print(df.shape)
    df.info()
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    return df.drop_duplicates()


def preprocess(df):
    df = df.copy()
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df = df.drop(columns=["Car_Name", "Year"])
    df = pd.get_dummies(
        df,
        columns=["Fuel_Type", "Seller_Type", "Transmission"],
        drop_first=True,
        dtype=int,
    )
    print(df.head())
    return df


def main():
    df = preprocess(load_data())
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    y_pred = model.predict(X_test)
    print("R2 Score:", round(r2_score(y_test, y_pred) * 100, 2), "%")
    print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

    results = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})
    print(results.head(10))

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Sample car, built by column name so the order always matches the model.
    # Fuel/Seller/Transmission columns are dummies (CNG, Dealer, Automatic = baseline).
    sample = {
        "Present_Price": 5.59,
        "Driven_kms": 27000,
        "Owner": 0,
        "Car_Age": 7,
        "Fuel_Type_Petrol": 1,
        "Seller_Type_Individual": 0,
        "Transmission_Manual": 1,
    }
    new_car = pd.DataFrame([sample]).reindex(columns=X.columns, fill_value=0)
    price = model.predict(new_car)[0]
    print("Predicted Car Selling Price:", round(price, 2), "Lakhs")


if __name__ == "__main__":
    main()
