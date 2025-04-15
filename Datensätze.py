import pandas as pd
import numpy as np

# Einlesen
df_2022 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2022.csv")
df_2023 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2023.csv")
df_2024 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2024.csv")

# Columns die gleich sein müssen, um eine Verbindung zu ergeben
id_cols = ['AIRLINE_ID', 'UNIQUE_CARRIER_ENTITY', 'ORIGIN', 'DEST', 'AIRCRAFT_TYPE', 'MONTH']

# Verbindung_id
df_2022['verbindung_id'] = df_2022['AIRLINE_ID'].astype(str) + '_' + df_2022['UNIQUE_CARRIER_ENTITY'].astype(str) + '_' + df_2022['ORIGIN'].astype(str) + '_' + df_2022['DEST'].astype(str) + '_' + df_2022['AIRCRAFT_TYPE'].astype(str) + '_' + df_2022['MONTH'].astype(str)
df_2023['verbindung_id'] = df_2023['AIRLINE_ID'].astype(str) + '_' + df_2023['UNIQUE_CARRIER_ENTITY'].astype(str) + '_' + df_2023['ORIGIN'].astype(str) + '_' + df_2023['DEST'].astype(str) + '_' + df_2023['AIRCRAFT_TYPE'].astype(str) + '_' + df_2023['MONTH'].astype(str)
df_2024['verbindung_id'] = df_2024['AIRLINE_ID'].astype(str) + '_' + df_2024['UNIQUE_CARRIER_ENTITY'].astype(str) + '_' + df_2024['ORIGIN'].astype(str) + '_' + df_2024['DEST'].astype(str) + '_' + df_2024['AIRCRAFT_TYPE'].astype(str) + '_' + df_2024['MONTH'].astype(str)

# verbindung_id und Jahr aus den Datensätzen filtern
df_2022_routes = df_2022[['verbindung_id', 'YEAR']].copy()
df_2023_routes = df_2023[['verbindung_id', 'YEAR']].copy()
df_2024_routes = df_2024[['verbindung_id', 'YEAR']].copy()

# Set erstellen wo jede verbindungs_id, die in allen 3 Jahren vorkommen, enthalten ist
common_routes = set(df_2022_routes['verbindung_id']) & set(df_2023_routes['verbindung_id']) & set(df_2024_routes['verbindung_id'])
common_routes = list(common_routes)

# Nur die common_routes behalten
df_2022_common = df_2022[df_2022['verbindung_id'].isin(common_routes)].copy()
df_2023_common = df_2023[df_2023['verbindung_id'].isin(common_routes)].copy()
df_2024_common = df_2024[df_2024['verbindung_id'].isin(common_routes)].copy()

# Verbinden zu neuem DF
merged_df = pd.concat([df_2022_common, df_2023_common, df_2024_common], ignore_index=True)


print(len(merged_df))

#'DEPARTURES_PERFORMED'=0 raus
merged_df = merged_df[merged_df['DEPARTURES_PERFORMED'] > 0].copy()

# p = Passagiere pro Flug
merged_df['p'] = np.floor(merged_df['PASSENGERS'] / merged_df['DEPARTURES_PERFORMED'])

# Filtern, sodass p >= k
k = 100
filtered_df = merged_df[merged_df['p'] >= k].copy()

print(len(filtered_df))

count_flights = (int(len(filtered_df))/36)
print(count_flights)
