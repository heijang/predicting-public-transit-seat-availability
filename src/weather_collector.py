"""Weather Data Collector

Collects real-time weather data for Berlin, Germany using Open-Meteo API.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Berlin coordinates
BERLIN_LAT = 52.52
BERLIN_LON = 13.405

# Open-Meteo API (free, no API key required)
WEATHER_API_URL = 'https://api.open-meteo.com/v1/forecast'

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'realtime' / 'weather'


# =============================================================================
# Core Functions
# =============================================================================

def fetch_weather_data(lat: float = BERLIN_LAT, lon: float = BERLIN_LON,
                       timeout: int = 30) -> dict:
    """Fetch current weather data from Open-Meteo API."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'current': [
            'temperature_2m',
            'relative_humidity_2m',
            'apparent_temperature',
            'precipitation',
            'rain',
            'snowfall',
            'weather_code',
            'cloud_cover',
            'wind_speed_10m',
            'wind_direction_10m',
        ],
        'daily': [
            'temperature_2m_max',
            'temperature_2m_min',
            'precipitation_sum',
            'rain_sum',
            'snowfall_sum',
        ],
        'timezone': 'Europe/Berlin',
        'forecast_days': 1,
    }

    response = requests.get(WEATHER_API_URL, params=params, timeout=timeout)
    response.raise_for_status()

    return response.json()


def parse_weather_response(data: dict) -> dict:
    """Parse weather API response into a flat record."""
    current = data.get('current', {})
    daily = data.get('daily', {})

    fetch_time = datetime.now()

    record = {
        'fetch_time': fetch_time,
        'location': 'Berlin',
        'latitude': data.get('latitude'),
        'longitude': data.get('longitude'),
        'timezone': data.get('timezone'),

        # Current weather
        'current_time': current.get('time'),
        'temperature': current.get('temperature_2m'),
        'apparent_temperature': current.get('apparent_temperature'),
        'humidity': current.get('relative_humidity_2m'),
        'precipitation': current.get('precipitation'),
        'rain': current.get('rain'),
        'snowfall': current.get('snowfall'),
        'weather_code': current.get('weather_code'),
        'cloud_cover': current.get('cloud_cover'),
        'wind_speed': current.get('wind_speed_10m'),
        'wind_direction': current.get('wind_direction_10m'),

        # Daily min/max
        'temperature_max': daily.get('temperature_2m_max', [None])[0],
        'temperature_min': daily.get('temperature_2m_min', [None])[0],
        'precipitation_sum': daily.get('precipitation_sum', [None])[0],
        'rain_sum': daily.get('rain_sum', [None])[0],
        'snowfall_sum': daily.get('snowfall_sum', [None])[0],
    }

    return record


def get_weather_description(weather_code: int) -> str:
    """Convert WMO weather code to description."""
    weather_codes = {
        0: 'Clear sky',
        1: 'Mainly clear',
        2: 'Partly cloudy',
        3: 'Overcast',
        45: 'Fog',
        48: 'Depositing rime fog',
        51: 'Light drizzle',
        53: 'Moderate drizzle',
        55: 'Dense drizzle',
        61: 'Slight rain',
        63: 'Moderate rain',
        65: 'Heavy rain',
        71: 'Slight snow',
        73: 'Moderate snow',
        75: 'Heavy snow',
        80: 'Slight rain showers',
        81: 'Moderate rain showers',
        82: 'Violent rain showers',
        85: 'Slight snow showers',
        86: 'Heavy snow showers',
        95: 'Thunderstorm',
        96: 'Thunderstorm with slight hail',
        99: 'Thunderstorm with heavy hail',
    }
    return weather_codes.get(weather_code, 'Unknown')


def save_to_daily_csv(record: dict, output_dir: Path,
                      fetch_time: datetime) -> Path:
    """Save weather record to daily partitioned CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    date_str = fetch_time.strftime('%Y%m%d')
    output_file = output_dir / f'weather_{date_str}.csv'

    df = pd.DataFrame([record])

    if output_file.exists():
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

    return output_file


def save_latest(record: dict, output_dir: Path) -> Path:
    """Save latest weather snapshot."""
    os.makedirs(output_dir, exist_ok=True)

    # Add weather description
    record = record.copy()
    record['weather_description'] = get_weather_description(
        record.get('weather_code', 0)
    )

    df = pd.DataFrame([record])
    latest_file = output_dir / 'latest_weather.csv'
    df.to_csv(latest_file, index=False)

    return latest_file


# =============================================================================
# Main Task Function (called by Airflow)
# =============================================================================

def collect_weather_data(output_dir: Path = None) -> dict:
    """
    Main collection task: fetch and save weather data.

    Returns:
        dict with collection statistics
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Fetch weather data
    data = fetch_weather_data()
    fetch_time = datetime.now()

    print(f"Fetch time: {fetch_time}")

    # Parse response
    record = parse_weather_response(data)

    print(f"Temperature: {record['temperature']}°C")
    print(f"Max: {record['temperature_max']}°C, Min: {record['temperature_min']}°C")
    print(f"Precipitation: {record['precipitation']}mm")
    print(f"Weather: {get_weather_description(record.get('weather_code', 0))}")

    # Save daily CSV
    output_file = save_to_daily_csv(record, output_dir, fetch_time)
    print(f"Saved to {output_file}")

    # Save latest snapshot
    latest_file = save_latest(record, output_dir)
    print(f"Saved latest to {latest_file}")

    return {
        'status': 'success',
        'temperature': record['temperature'],
        'output_file': str(output_file),
        'fetch_time': fetch_time.isoformat()
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    result = collect_weather_data()
    print(f"Result: {result}")
