import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

# Load list of connections to evaluate
connections = pd.read_csv("passed_connections.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

data = pd.read_csv("verbindungen_mit_kennzahlen.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH']].assign(DAY=1))
data.sort_values('DATE', inplace=True)

# Parameter grid for Holt-Winters
trend_opts = ['add', 'mul']
seasonal_opts = ['add', 'mul']
damped_opts = [True, False]
params_grid = list(product(trend_opts, seasonal_opts, damped_opts))

# Helper to evaluate one split

def evaluate_split(train, test, trend, seasonal, damped):
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12,
        damped_trend=damped,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = model.forecast(len(test))
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)
    return mae, rmse, r2

# Cross validation splits - two folds (2022->2023 and 2022+2023->2024)

def connection_splits(df_conn):
    df_conn = df_conn.set_index("DATE").sort_index()
    train1 = df_conn[df_conn.index.year == 2022]["PASSENGERS"]
    test1 = df_conn[df_conn.index.year == 2023]["PASSENGERS"]
    train2 = df_conn[df_conn.index.year <= 2023]["PASSENGERS"]
    test2 = df_conn[df_conn.index.year == 2024]["PASSENGERS"]
    if len(train1) >= 12 and len(test1) >= 12 and len(train2) >= 24 and len(test2) >= 12:
        return [(train1, test1), (train2, test2)]
    return []

results = []

for trend, seasonal, damped in params_grid:
    fold_metrics = []
    for _, conn in connections.iterrows():
        mask = (
            (data["AIRLINE_ID"] == conn["AIRLINE_ID"]) &
            (data["UNIQUE_CARRIER_ENTITY"] == conn["UNIQUE_CARRIER_ENTITY"]) &
            (data["ORIGIN"] == conn["ORIGIN"]) &
            (data["DEST"] == conn["DEST"]) &
            (data["AIRCRAFT_TYPE"] == conn["AIRCRAFT_TYPE"])
        )
        df_conn = data[mask]
        splits = connection_splits(df_conn)
        for train, test in splits:
            try:
                mae, rmse, r2 = evaluate_split(train, test, trend, seasonal, damped)
                fold_metrics.append((mae, rmse, r2))
            except Exception:
                pass
    if fold_metrics:
        arr = np.array(fold_metrics)
        results.append({
            "trend": trend,
            "seasonal": seasonal,
            "damped": damped,
            "MAE": arr[:,0].mean(),
            "RMSE": arr[:,1].mean(),
            "R2": arr[:,2].mean(),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["RMSE"]).reset_index(drop=True)
print("Mean metrics over all connections per parameter combination:")
print(results_df)

best = results_df.iloc[0]
print("\nBest parameter set:")
print(best)
results_df.to_csv("holt_winters_cv_results.csv", index=False)