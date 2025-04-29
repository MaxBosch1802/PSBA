#! /usr/bin/python3
import os, sys
import pandas as pd
import math

def check_connection(con, df, k):

    air, uce, org, dst, atp = con
    allmonths = [1, 2, 3, 4,5, 6,7, 8,9, 10,11,12]

    subdf = df[(df["AIRLINE_ID"] == air)]
    subdf = subdf[(subdf["UNIQUE_CARRIER_ENTITY"] == uce)]
    subdf = subdf[(subdf["ORIGIN"] == org)]
    subdf = subdf[(subdf["DEST"] == dst)]
    subdf = subdf[(subdf["AIRCRAFT_TYPE"] == atp)]

    months = [int(a) for a in subdf["MONTH"]]

    if len(months) < 12:
        return False

    ok = True

    for mon in allmonths:
        if (mon not in months):
            ok = False
            break

    if (ok == False):
        return ok

    for mon in allmonths:
        numpax = 0
        dfsel = subdf[(subdf["MONTH"] == mon)]
        pax = dfsel["PASSENGERS"]
        dep = dfsel["DEPARTURES_PERFORMED"]
        for r in range(0, pax.size):
            if (int(dep.iat[r]) > 0):
                #numpax += int (int(pax.iat[r]) / int(dep.iat[r]))
                numpax += math.ceil((int(pax.iat[r]) / int(dep.iat[r])))
        if numpax < k:
            ok = False
            break

    if (ok == True):
        print(con, "passed")

    return ok

def print_connection(con, df):
    air, uce, org, dst, atp = con

    subdf = df[(df["AIRLINE_ID"] == air)]
    subdf = subdf[(subdf["UNIQUE_CARRIER_ENTITY"] == uce)]
    subdf = subdf[(subdf["ORIGIN"] == org)]
    subdf = subdf[(subdf["DEST"] == dst)]
    subdf = subdf[(subdf["AIRCRAFT_TYPE"] == atp)]

    print(subdf)

def read(csv1, csv2, csv3):
    df1 = pd.read_csv(csv1, sep=",", usecols=["PASSENGERS", "DEPARTURES_PERFORMED", "SEATS", "AIRLINE_ID","UNIQUE_CARRIER_ENTITY","ORIGIN","DEST","AIRCRAFT_TYPE","MONTH"])
    df2 = pd.read_csv(csv2, sep=",", usecols=["PASSENGERS", "DEPARTURES_PERFORMED", "SEATS", "AIRLINE_ID","UNIQUE_CARRIER_ENTITY","ORIGIN","DEST","AIRCRAFT_TYPE","MONTH"])
    df3 = pd.read_csv(csv3, sep=",", usecols=["PASSENGERS", "DEPARTURES_PERFORMED", "SEATS", "AIRLINE_ID","UNIQUE_CARRIER_ENTITY","ORIGIN","DEST","AIRCRAFT_TYPE","MONTH"])

    k = 100

    connections = set()
    passed = []

    for index, row in df1.iterrows():
        air = row["AIRLINE_ID"]
        uce = row["UNIQUE_CARRIER_ENTITY"]
        org = row["ORIGIN"]
        dst = row["DEST"]
        atp = row["AIRCRAFT_TYPE"]

        con = air, uce, org, dst, atp

        connections.add(con)

    print("Found",len(connections), " connections in first data frame.")

    for con in connections:

        result = check_connection(con, df1, k)

        if (result == True):
            result = check_connection(con, df2, k)

        if (result == True):
            result = check_connection(con, df3, k)

        if (result == True):
            passed.append(con)

    print(len(passed), "connections passed for all data frames.")
    
    df_passed = pd.DataFrame(passed, columns=[
        "AIRLINE_ID", 
        "UNIQUE_CARRIER_ENTITY", 
        "ORIGIN", 
        "DEST", 
        "AIRCRAFT_TYPE"
    ])
    df_passed.to_csv("passed_connections.csv", index=False)


    
if __name__ == "__main__":
    csv1='T_T100I_SEGMENT_ALL_CARRIER_2022.csv'
    csv2='T_T100I_SEGMENT_ALL_CARRIER_2023.csv'
    csv3='T_T100I_SEGMENT_ALL_CARRIER_2024.csv'
    read(csv1, csv2, csv3)


