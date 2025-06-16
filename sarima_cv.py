import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

# Load list of connections to evaluate
connections = pd.read_csv("passed_connections.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# Load main dataset
data = pd.read_csv("verbindungen_mit_kennzahlen.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# Prepare date column and sorting
data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH']].assign(DAY=1))
data.sort_values('DATE', inplace=True)

# Parameter grid for SARIMA (order) and seasonal_order with seasonal period 12
p_values = [0, 1]
d_values = [0, 1]
q_values = [0, 1]
P_values = [0, 1]
D = 1
Q_values = [0, 1]
params_grid = list(product(p_values, d_values, q_values, P_values, Q_values))


def evaluate_split(train, test, order, seasonal_order):
    """Fit SARIMA on train and forecast the length of test."""
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = model.forecast(len(test))
        if forecast.isna().any() or test.isna().any():
            return None
    except Exception:
        return None
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)
    return mae, rmse, r2


# Create two train/test splits

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

for p, d, q, P, Q in params_grid:
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
            metrics = evaluate_split(train, test, (p, d, q), (P, D, Q, 12))
            if metrics:
                fold_metrics.append(metrics)
    if fold_metrics:
        arr = np.array(fold_metrics)
        results.append({
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "MAE": arr[:, 0].mean(),
            "RMSE": arr[:, 1].mean(),
            "R2": arr[:, 2].mean(),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["RMSE"]).reset_index(drop=True)

print("Mean metrics over all connections per parameter combination:")
print(results_df)

best = results_df.iloc[0]
print("\nBest parameter set:")
print(best)

results_df.to_csv("sarima_cv_results.csv", index=False)