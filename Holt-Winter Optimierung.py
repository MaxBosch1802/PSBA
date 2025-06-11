import itertools
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

# # 4. NUR RELEVANTE SPALTEN
# ts_df = route_df[['DATE', 'PASSENGERS']].copy()
# ts_df = ts_df.groupby('DATE').sum().reset_index()

# # 5. TRAIN/TEST SPLIT
# split_date = '2024-01-01'
# train = ts_df[ts_df['DATE'] < split_date]
# test = ts_df[ts_df['DATE'] >= split_date]

# # 6. METRIKEN-FUNKTION
# def evaluate_forecast(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)
#     return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# #Alle Parameterkombinationen definieren
# trend_options = ['add', 'mul']
# seasonal_options = ['add', 'mul']
# damped_options = [True, False]
# boxcox_options = [True, False]

# # Train und Test Daten
# y_train = train['PASSENGERS']
# y_test = test['PASSENGERS']

# # Ergebnisse speichern
# results = []

# # Grid Search
# for trend, seasonal, damped, boxcox in itertools.product(trend_options, seasonal_options, damped_options, boxcox_options):
#     try:
#         model = ExponentialSmoothing(
#             y_train,
#             trend=trend,
#             seasonal=seasonal,
#             seasonal_periods=12,
#             damped_trend=damped,
#             use_boxcox=boxcox
#         ).fit(optimized=True)
#         forecast = model.forecast(len(y_test))
#         mae = mean_absolute_error(y_test, forecast)
#         rmse = np.sqrt(mean_squared_error(y_test, forecast))
#         r2 = r2_score(y_test, forecast)

#         results.append({
#             'Trend': trend,
#             'Seasonal': seasonal,
#             'Damped': damped,
#             'BoxCox': boxcox,
#             'MAE': mae,
#             'RMSE': rmse,
#             'R2': r2
#         })
#     except Exception as e:
#         print(f"Fehler bei Kombination {trend}-{seasonal}-{damped}-{boxcox}: {e}")

# # Ergebnisse sortieren
# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by='RMSE')
# print(results_df)


import pandas as pd

# 1. Aggregiere alle Passagiere pro Monat
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# Summe pro Monat
agg_df = df.groupby('DATE')['PASSENGERS'].sum().reset_index()

# 2. Train/Test Split
train = agg_df[agg_df['DATE'] < '2024-01-01']
test = agg_df[agg_df['DATE'] >= '2024-01-01']

y_train = train['PASSENGERS']
y_test = test['PASSENGERS']

# 3. Holt-Winters Modell auf Summen
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Hier nimmst du deine vorher beste Konfiguration:
model = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=True,
    use_boxcox=False
).fit(optimized=True)

# Vorhersage
forecast = model.forecast(len(y_test))

# Fehlermaße
mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))
r2 = r2_score(y_test, forecast)

print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")

# 4. Plotten
import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))
plt.plot(train['DATE'], y_train, label='Train')
plt.plot(test['DATE'], y_test, label='Test', color='black')
plt.plot(test['DATE'], forecast, label='Holt-Winters Forecast (aggregiert)', linestyle='--')
plt.legend()
plt.title('Forecast of Total Monthly Passenger Numbers')
plt.xlabel('Date')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.show()
