import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# === CSV-Dateien laden ===
df_passed = pd.read_csv("passed_connections.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})
df_data = pd.read_csv("verbindungen_mit_kennzahlen.csv", dtype={"UNIQUE_CARRIER_ENTITY": str})

# === Zeitachse erzeugen ===
df_data["DATE"] = pd.to_datetime(df_data["YEAR"].astype(str) + "-" + df_data["MONTH"].astype(str).str.zfill(2) + "-01")
df_data.sort_values("DATE", inplace=True)

# === Ergebnisliste vorbereiten ===
results = []

# === Über alle gültigen Verbindungen iterieren ===
for _, row in df_passed.iterrows():
    verbindung = {
        "AIRLINE_ID": row["AIRLINE_ID"],
        "UNIQUE_CARRIER_ENTITY": row["UNIQUE_CARRIER_ENTITY"],
        "ORIGIN": row["ORIGIN"],
        "DEST": row["DEST"],
        "AIRCRAFT_TYPE": row["AIRCRAFT_TYPE"]
    }

    # Verbindung filtern
    mask = (
        (df_data["AIRLINE_ID"] == verbindung["AIRLINE_ID"]) &
        (df_data["UNIQUE_CARRIER_ENTITY"] == verbindung["UNIQUE_CARRIER_ENTITY"]) &
        (df_data["ORIGIN"] == verbindung["ORIGIN"]) &
        (df_data["DEST"] == verbindung["DEST"]) &
        (df_data["AIRCRAFT_TYPE"] == verbindung["AIRCRAFT_TYPE"])
    )
    df_verbindung = df_data[mask].copy()

    # Trainingsdaten (2022–2023)
    train = df_verbindung[df_verbindung["YEAR"] < 2024].set_index("DATE")["PASSENGERS"]
    test = df_verbindung[df_verbindung["YEAR"] == 2024].set_index("DATE")["PASSENGERS"]

    # Nur vollständige Reihen verarbeiten
    if train.isna().any() or len(train) < 24 or len(test) < 12:
        continue

    try:
        model = ExponentialSmoothing(
            train,
            seasonal="add",
            trend="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast = fit.forecast(12)

        # Vergleich mit echten Werten
        common_index = forecast.index.intersection(test.index)
        y_true = test.loc[common_index]
        y_pred = forecast.loc[common_index]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        results.append({
            **verbindung,
            "RMSE": rmse,
            "MAPE": mape
        })

    except Exception as e:
        # Optional: Logging oder Fehlerausgabe
        print(f"⚠️ Fehler bei Verbindung {verbindung}: {e}")
        continue

# === Ergebnisse als DataFrame ===
df_results = pd.DataFrame(results)

# === Ausgabe anzeigen ===
print(df_results.head())
print(f"\n✅ Gesamt: {len(df_results)} Verbindungen erfolgreich ausgewertet.")


# === Monatliche Gesamtwerte sammeln ===
gesamt_forecasts = []
gesamt_true = []

# Nochmals über die Verbindungen gehen und Forecasts speichern
for _, row in df_passed.iterrows():
    verbindung = {
        "AIRLINE_ID": row["AIRLINE_ID"],
        "UNIQUE_CARRIER_ENTITY": row["UNIQUE_CARRIER_ENTITY"],
        "ORIGIN": row["ORIGIN"],
        "DEST": row["DEST"],
        "AIRCRAFT_TYPE": row["AIRCRAFT_TYPE"]
    }

    mask = (
        (df_data["AIRLINE_ID"] == verbindung["AIRLINE_ID"]) &
        (df_data["UNIQUE_CARRIER_ENTITY"] == verbindung["UNIQUE_CARRIER_ENTITY"]) &
        (df_data["ORIGIN"] == verbindung["ORIGIN"]) &
        (df_data["DEST"] == verbindung["DEST"]) &
        (df_data["AIRCRAFT_TYPE"] == verbindung["AIRCRAFT_TYPE"])
    )
    df_verbindung = df_data[mask].copy()

    train = df_verbindung[df_verbindung["YEAR"] < 2024].set_index("DATE")["PASSENGERS"]
    test = df_verbindung[df_verbindung["YEAR"] == 2024].set_index("DATE")["PASSENGERS"]

    if train.isna().any() or len(train) < 24 or len(test) < 12:
        continue

    try:
        model = ExponentialSmoothing(
            train,
            seasonal="add",
            trend="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast = fit.forecast(12)

        # Speichern von Prognose + echten Werten
        forecast_df = pd.DataFrame({
            "DATE": forecast.index,
            "FORECAST": forecast.values
        })

        test_df = pd.DataFrame({
            "DATE": test.index,
            "TRUE": test.values
        })

        merged = pd.merge(forecast_df, test_df, on="DATE", how="inner")
        gesamt_forecasts.append(merged)

    except:
        continue

# === Alle Prognosen & echten Werte zusammenführen ===
df_gesamt = pd.concat(gesamt_forecasts)
df_monatlich = df_gesamt.groupby("DATE").sum()


rmse_gesamt = np.sqrt(mean_squared_error(df_monatlich["TRUE"], df_monatlich["FORECAST"]))
mape_gesamt = np.mean(np.abs((df_monatlich["TRUE"] - df_monatlich["FORECAST"]) / df_monatlich["TRUE"])) * 100

# === Ergebnis ausgeben ===
print("\n📊 Gesamter Fehler über alle Verbindungen:")
print(f"✅ Gesamt-RMSE: {rmse_gesamt:.2f} Passagiere")
print(f"✅ Gesamt-MAPE: {mape_gesamt:.2f}%")
