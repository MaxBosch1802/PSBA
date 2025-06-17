import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

# Load connections
connections = pd.read_csv("passed_connections.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# Load main dataset
data = pd.read_csv("verbindungen_mit_kennzahlen.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH']].assign(DAY=1))
data.sort_values('DATE', inplace=True)

# Parameter grid for polynomial degree
degrees = [1, 2, 3]
params_grid = degrees


def evaluate_split(train, test, degree):
    """Fit polynomial regression on train and forecast the length of test."""
    # create time index starting at 0 for train start
    start_date = train.index.min()
    train_t = (train.index - start_date).days.values.reshape(-1, 1)
    test_t = (test.index - start_date).days.values.reshape(-1, 1)
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train = poly.fit_transform(train_t)
    X_test = poly.transform(test_t)
    model = LinearRegression()
    model.fit(X_train, train.values)
    forecast = model.predict(X_test)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)
    return mae, rmse, r2


def connection_splits(df_conn):
    """Return two train/test splits if enough data is available."""
    df_conn = df_conn.set_index('DATE').sort_index()
    train1 = df_conn[df_conn.index.year == 2022]['PASSENGERS']
    test1 = df_conn[df_conn.index.year == 2023]['PASSENGERS']
    train2 = df_conn[df_conn.index.year <= 2023]['PASSENGERS']
    test2 = df_conn[df_conn.index.year == 2024]['PASSENGERS']
    if len(train1) >= 12 and len(test1) >= 12 and len(train2) >= 24 and len(test2) >= 12:
        return [(train1, test1), (train2, test2)]
    return []


results = []

for degree in params_grid:
    fold_metrics = []
    for _, conn in connections.iterrows():
        mask = (
            (data['AIRLINE_ID'] == conn['AIRLINE_ID']) &
            (data['UNIQUE_CARRIER_ENTITY'] == conn['UNIQUE_CARRIER_ENTITY']) &
            (data['ORIGIN'] == conn['ORIGIN']) &
            (data['DEST'] == conn['DEST']) &
            (data['AIRCRAFT_TYPE'] == conn['AIRCRAFT_TYPE'])
        )
        df_conn = data[mask]
        splits = connection_splits(df_conn)
        for train, test in splits:
            try:
                metrics = evaluate_split(train, test, degree)
                fold_metrics.append(metrics)
            except Exception:
                pass
    if fold_metrics:
        arr = np.array(fold_metrics)
        results.append({
            'degree': degree,
            'MAE': arr[:, 0].mean(),
            'RMSE': arr[:, 1].mean(),
            'R2': arr[:, 2].mean(),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['RMSE']).reset_index(drop=True)

print("Mean metrics over all connections per parameter combination:")
print(results_df)

best = results_df.iloc[0]
print("\nBest parameter set:")
print(best)

results_df.to_csv("linear_regression_cv_results.csv", index=False)
