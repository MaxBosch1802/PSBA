import pandas as pd
import numpy as np

# CSV einlesen
df_2022 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2022.csv")
df_2023 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2023.csv")
df_2024 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2024.csv")


df_all = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

# Jahr-Monat kombinieren zu einer neuen Spalte
df_all["YEAR_MONTH"] = df_all["YEAR"].astype(str) + "-" + df_all["MONTH"].astype(str).str.zfill(2)

# wenn group_cols gleich, dann gleiche Verbindung
group_cols = [
    "AIRLINE_ID",
    "UNIQUE_CARRIER_ENTITY",
    "ORIGIN",
    "DEST",
    "AIRCRAFT_TYPE"
]

# Gruppieren nach Verbindung und zählen der unique Jahr-Monat-Kombis
grouped = df_all.groupby(group_cols)["YEAR_MONTH"].nunique().reset_index()

# Nur Verbindungen mit 36 vollständigen Monatseinträgen (12 * 3 Jahre) behalten
valid_routes = grouped[grouped["YEAR_MONTH"] == 36]

#Gefilterte Daten extrahieren
df_filtered = df_all.merge(valid_routes[group_cols], on=group_cols, how="inner")


#'DEPARTURES_PERFORMED'=0 raus
#df_filtered = df_filtered[df_filtered['DEPARTURES_PERFORMED'] > 0].copy()

# p = Passagiere pro Flug
df_filtered['p'] = np.floor(df_filtered['PASSENGERS'] / df_filtered['DEPARTURES_PERFORMED'])

k = 100
df_passagiere = df_filtered[df_filtered['p'] >= k].copy()

print("Anzahl Zeilen im Datensatz")
print(len(df_passagiere))

count_flights = (int(len(df_passagiere))/36)
print("Anzahl verbindungen")
print(count_flights)

# Anzahl der eindeutigen Verbindungen nach dem Filtern
anzahl_verbindungen = df_passagiere[group_cols].drop_duplicates().shape[0]
print(anzahl_verbindungen)

print(grouped.info)

grouped.to_csv("grouped.csv", index=False)

df_filtered.to_csv("df_filtered.csv", index=False)