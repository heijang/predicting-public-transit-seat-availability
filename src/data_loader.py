import pandas as pd
import os

def load_occupancies(filepath='data/raw/occupancies_20241008.csv', output_dir='data/ingested'):
    df = pd.read_csv(filepath, sep=',')
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])

    os.makedirs(output_dir, exist_ok=True)

    re5_path = os.path.join(output_dir, 'RE5_2024_03.csv')
    re6_path = os.path.join(output_dir, 'RE6_2024_03.csv')

    # re5 = df[(df['line'] == 'RE5') & (df['derived_capacity'] == 'LWD')].copy()
    # re6 = df[(df['line'] == 'RE6') & (df['derived_capacity'] == 'LWD')].copy()
    re5 = df[df['line'] == 'RE5'].copy()
    re6 = df[df['line'] == 'RE6'].copy()

    re5.to_csv(re5_path, index=False)
    re6.to_csv(re6_path, index=False)
    
    return re5, re6

def load_matches(filepath='data/raw/Schedule_Teams_2022_2024.xlsx'):
    return pd.read_excel(filepath)


def load_stops(filepath='data/raw/stops.txt'):
    return pd.read_csv(filepath)


def main():
    raw_data = load_occupancies()
    print(raw_data)


if __name__ == "__main__":
    main()