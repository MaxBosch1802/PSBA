import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

# Load list of connections
connections = pd.read_csv("passed_connections.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# Load main dataset
data = pd.read_csv("verbindungen_mit_kennzahlen.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# Prepare date column and sort
data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH']].assign(DAY=1))
data.sort_values('DATE', inplace=True)

# Parameter grid
# Reduced grid to shorten runtime
cp_scales = [0.05, 0.1]
seasonality_modes = ['additive']
params_grid = list(product(cp_scales, seasonality_modes))


def evaluate_split(train_df, test_df, cp_scale, seasonality_mode):
    """Fit Prophet on train and forecast the length of test."""
    train_p = train_df.rename(columns={'DATE': 'ds', 'PASSENGERS': 'y'})
    model = Prophet(
        changepoint_prior_scale=cp_scale,
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
    )
    model.fit(train_p)
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = model.predict(future)
    fc = forecast['yhat'].iloc[-len(test_df):].values
    mae = mean_absolute_error(test_df['PASSENGERS'], fc)
    rmse = np.sqrt(mean_squared_error(test_df['PASSENGERS'], fc))
    r2 = r2_score(test_df['PASSENGERS'], fc)
    return mae, rmse, r2


def connection_splits(df_conn):
    """Return two train/test splits if enough data is available."""
    df_conn = df_conn.set_index('DATE').sort_index()
    train1 = df_conn[df_conn.index.year == 2022]['PASSENGERS'].reset_index()
    test1 = df_conn[(df_conn.index >= '2023-01-01') & (df_conn.index < '2023-07-01')]['PASSENGERS'].reset_index()
    train2 = df_conn[df_conn.index < '2024-01-01']['PASSENGERS'].reset_index()
    test2 = df_conn[(df_conn.index >= '2024-01-01') & (df_conn.index < '2024-07-01')]['PASSENGERS'].reset_index()
    if len(train1) >= 12 and len(test1) >= 6 and len(train2) >= 24 and len(test2) >= 6:
        return [(train1, test1), (train2, test2)]
    return []


results = []

for cp_scale, seasonality_mode in params_grid:
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
                metrics = evaluate_split(train, test, cp_scale, seasonality_mode)
                fold_metrics.append(metrics)
            except Exception:
                pass
    if fold_metrics:
        arr = np.array(fold_metrics)
        results.append({
            'changepoint_prior_scale': cp_scale,
            'seasonality_mode': seasonality_mode,
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

results_df.to_csv("prophet_cv_results.csv", index=False)
