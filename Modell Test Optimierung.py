# WICHTIGE MODULE
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. DATEN LADEN
df = pd.read_csv("verbindungen_mit_kennzahlen.csv")
df['ROUTE'] = df['ORIGIN'] + '-' + df['DEST']

# 2. DATUMSSPALTE ERSTELLEN
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# 3. FILTERN NACH ROUTE
route_df = df[df['ROUTE'] == 'JFK-LHR'].sort_values('DATE')

# 4. NUR RELEVANTE SPALTEN
ts_df = route_df[['DATE', 'PASSENGERS']].copy()
ts_df = ts_df.groupby('DATE').sum().reset_index()

# 5. TRAIN/TEST SPLIT
split_date = '2024-01-01'
train = ts_df[ts_df['DATE'] < split_date]
test = ts_df[ts_df['DATE'] >= split_date]

# 6. METRIKEN-FUNKTION
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# 7. HOLT-WINTERS MODELL
def forecast_holt_winters(train, test, seasonal_periods=12):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods,damped_trend =True, use_boxcox=False).fit()
    forecast = model.forecast(len(test))
    return forecast

# 8. ARIMA MODELL
def forecast_arima(train, test, order=(1,0,1)):
    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=len(test))
    return forecast

# 9. SARIMA MODELL
def forecast_sarima(train, test, order=(0,0,0), seasonal_order=(1,1,0,12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit()
    forecast = model.forecast(steps=len(test))
    return forecast

# 10. PROPHET MODELL
def forecast_prophet(train_df, test_df):
    prophet_df = train_df.rename(columns={'DATE': 'ds', 'PASSENGERS': 'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = model.predict(future)
    return forecast['yhat'][-len(test_df):].values

# 11. MODELLE TRAINIEREN UND PROGNOSTIZIEREN
hw_forecast = forecast_holt_winters(train['PASSENGERS'], test)
arima_forecast = forecast_arima(train['PASSENGERS'], test)
sarima_forecast = forecast_sarima(train['PASSENGERS'], test)
prophet_forecast = forecast_prophet(train, test)

# 12. METRIKEN BERECHNEN
hw_metrics = evaluate_forecast(test['PASSENGERS'], hw_forecast)
arima_metrics = evaluate_forecast(test['PASSENGERS'], arima_forecast)
sarima_metrics = evaluate_forecast(test['PASSENGERS'], sarima_forecast)
prophet_metrics = evaluate_forecast(test['PASSENGERS'], prophet_forecast)

# 13. METRIKEN VERGLEICH
metrics_df = pd.DataFrame({
    'Holt-Winters': hw_metrics,
    'ARIMA': arima_metrics,
    'SARIMA': sarima_metrics,
    'Prophet': prophet_metrics
})
print(metrics_df)

# 14. VISUALISIERUNG
plt.figure(figsize=(14,7))
plt.plot(train['DATE'], train['PASSENGERS'], label='Train')
plt.plot(test['DATE'], test['PASSENGERS'], label='Test', color='black')
plt.plot(test['DATE'], hw_forecast, label='Holt-Winters Forecast')
plt.plot(test['DATE'], arima_forecast, label='ARIMA Forecast')
plt.plot(test['DATE'], sarima_forecast, label='SARIMA Forecast')
plt.plot(test['DATE'], prophet_forecast, label='Prophet Forecast')
plt.legend()
plt.title('Forecast vs Actuals - JFK to LHR')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.grid(True)
plt.show()
