import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# CSV einlesen
df = pd.read_csv("aggregierte_verbindungen.csv")

# Neue Spalte: Passagiere pro Flug (abgerundet)
df["PAX_PRO_FLUG"] = np.floor(df["PASSENGERS"] / df["DEPARTURES_PERFORMED"])

# Optional: Ergebnis anzeigen
print(df[["PASSENGERS", "DEPARTURES_PERFORMED", "PAX_PRO_FLUG"]].head())

# Optional: Zwischenspeichern
df.to_csv("verbindungen_mit_pax_pro_flug.csv", index=False)

print("✅ Passagiere pro Flug berechnet und gespeichert.")

# Neue Spalte: Auslastungsgrad berechnen
df["AUSLASTUNG"] = df["PASSENGERS"] / df["SEATS"]

# Optional: Begrenzung (falls Auslastung > 1 durch fehlerhafte Daten)
df["AUSLASTUNG"] = df["AUSLASTUNG"].clip(upper=1.0)

# Vorschau anzeigen
print(df[["PASSENGERS", "SEATS", "AUSLASTUNG"]].head())

# Ergebnis speichern
df.to_csv("verbindungen_mit_kennzahlen.csv", index=False)

print("✅ Auslastungsgrad berechnet und gespeichert.")

