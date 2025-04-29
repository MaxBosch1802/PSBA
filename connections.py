import pandas as pd


df_2022 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2022.csv")
df_2023 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2023.csv")
df_2024 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2024.csv")


df_all = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)


df_passed = pd.read_csv("passed_connections.csv")

group_cols = [
    "AIRLINE_ID",
    "UNIQUE_CARRIER_ENTITY",
    "ORIGIN",
    "DEST",
    "AIRCRAFT_TYPE"
]

df_filtered = df_all.merge(df_passed, on=group_cols, how="inner")


print(f"✅ Gefiltert: {len(df_filtered)} Zeilen in df_filtered.")

#df_filtered.to_csv("gefilterte_verbindungen.csv", index=False)

# Schritt 1: Definiere Gruppierungsspalten
agg_group_cols = [
    "AIRLINE_ID",
    "UNIQUE_CARRIER_ENTITY",
    "ORIGIN",
    "DEST",
    "AIRCRAFT_TYPE",
    "YEAR",
    "MONTH"
]

# Schritt 2: Wähle nur echte numerische Spalten für Aggregation
numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()

# Schritt 3: Entferne alle Spalten, die zum Gruppieren genutzt werden (besonders AIRCRAFT_TYPE, YEAR, MONTH)
numeric_cols = [col for col in numeric_cols if col not in agg_group_cols]

# Schritt 4: Jetzt sicher aggregieren
df_aggregated = df_filtered.groupby(agg_group_cols)[numeric_cols].sum().reset_index()

# Schritt 5: Ergebnis speichern oder anzeigen
df_aggregated.to_csv("aggregierte_verbindungen.csv", index=False)

print(f"✅ Aggregation abgeschlossen: {len(df_aggregated)} Zeilen im Ergebnis.")