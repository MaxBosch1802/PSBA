import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Datei einlesen
df = pd.read_csv("verbindungen_mit_kennzahlen.csv")

# Verbindung auswählen (Beispiel)
verbindung = {
    "AIRLINE_ID": 19393,
    "UNIQUE_CARRIER_ENTITY": "11033",  
    "ORIGIN": "CUN",
    "DEST": "BWI",
    "AIRCRAFT_TYPE": 612
}


# Nur Daten für diese Verbindung
mask = (df["AIRLINE_ID"] == verbindung["AIRLINE_ID"]) & \
       (df["UNIQUE_CARRIER_ENTITY"] == verbindung["UNIQUE_CARRIER_ENTITY"]) & \
       (df["ORIGIN"] == verbindung["ORIGIN"]) & \
       (df["DEST"] == verbindung["DEST"]) & \
       (df["AIRCRAFT_TYPE"] == verbindung["AIRCRAFT_TYPE"])

df_verbindung = df[mask].copy()

# Sortieren & Zeitachse erzeugen
df_verbindung["DATE"] = pd.to_datetime(df_verbindung["YEAR"].astype(str) + "-" + df_verbindung["MONTH"].astype(str) + "-01")
df_verbindung.sort_values("DATE", inplace=True)


# Zeitreihe (2022+2023) → Trainingsdaten
train = df_verbindung[df_verbindung["YEAR"] < 2024].set_index("DATE")["PASSENGERS"]


# Holt-Winters Modell fitten
model = ExponentialSmoothing(train, seasonal="add", trend="add", seasonal_periods=12)
fit = model.fit()

# Prognose für 12 Monate (2024)
forecast = fit.forecast(12)

# Vergleich mit echten 2024-Werten
test = df_verbindung[df_verbindung["YEAR"] == 2024].set_index("DATE")["PASSENGERS"]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train, label="Trainingsdaten (2022-2023)")
plt.plot(test, label="Echte Werte 2024")
plt.plot(forecast, label="Holt-Winters Prognose 2024", linestyle="--")
plt.title("Prognose Passagierzahlen – Beispielverbindung")
plt.xlabel("Monat")
plt.ylabel("Passagiere")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Fehlerberechnung
common_index = forecast.index.intersection(test.index)
y_true = test.loc[common_index]
y_pred = forecast.loc[common_index]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # in %

print(f"✅ RMSE: {rmse:.2f} Passagiere")
print(f"✅ MAPE: {mape:.2f}%")