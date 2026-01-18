import pandas as pd
import os

def load_occupancies(filepath='data/raw/occupancies_20241008.csv'):
    df = pd.read_csv(filepath, sep=',')
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])
    return df

def load_matches(filepath='data/raw/Schedule_Teams_2022_2024.xlsx'):
    return pd.read_excel(filepath)

def load_stops(filepath='data/raw/stops.txt'):
    return pd.read_csv(filepath)

