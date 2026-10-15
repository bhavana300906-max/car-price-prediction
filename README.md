# Car Price Prediction using Linear Regression

Predicts the selling price (in lakhs) of a used car from its present price, kilometres driven, age, owner count, fuel type, seller type and transmission.

## Features

- Data inspection and duplicate removal
- Feature engineering: `Car_Age` from `Year`
- One-hot encoding of categorical columns
- 80/20 train/test split, Linear Regression
- R² score and Mean Absolute Error
- Actual vs predicted scatter plot and a sample prediction

## Project Structure

```
car-price-prediction/
├── data/
│   └── car data.csv        # add the dataset here (not included)
├── car_price_prediction.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Download the "Vehicle dataset" / `car data.csv` (e.g. from Kaggle) and save it as `data/car data.csv`.

Columns: `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`.

## Installation

```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

## Usage

```bash
python car_price_prediction.py
```

Metrics are printed, the plot is saved as `actual_vs_predicted.png`, and a sample prediction is shown. Edit the `sample` dictionary in `main()` to predict for your own car.

## Notes

`Car_Age` uses the current year, so results shift slightly from year to year.

## License

MIT
