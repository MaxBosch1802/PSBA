import pandas as pd


df_2022 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2022.csv")
df_2023 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2023.csv")
df_2024 = pd.read_csv("T_T100I_SEGMENT_ALL_CARRIER_2024.csv")


df_all = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)


df_passed = pd.read_csv("passed_connections.csv")



