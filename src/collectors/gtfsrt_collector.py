"""DB Timetables API Collector

Collects RE5 train arrival information from Deutsche Bahn Timetables API.
"""

import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

DB_API_BASE_URL = 'https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1/plan'
DB_CLIENT_ID = '260da1dc7c1ad89c65b36b72bf7108dc'
DB_API_KEY = 'cb06900952ad02b2efd3fc015a04da8e'

# Berlin Hauptbahnhof station ID
BERLIN_HBF_STATION_ID = '8098160'

TARGET_LINES = ['RE5', 'RE6']
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'realtime' / 'gtfsrt'


# =============================================================================
# Core Functions
# =============================================================================

def fetch_timetable(station_id: str = BERLIN_HBF_STATION_ID,
                    date: datetime = None,
                    hour: int = None,
                    timeout: int = 30) -> str:
    """Fetch timetable XML from DB Timetables API.

    Args:
        station_id: Station EVA number
        date: Date to query (default: today)
        hour: Hour to query (default: current hour)
        timeout: Request timeout in seconds

    Returns:
        XML response string
    """
    if date is None:
        date = datetime.now()
    if hour is None:
        hour = datetime.now().hour

    # Format: YYMMDD
    date_str = date.strftime('%y%m%d')

    url = f"{DB_API_BASE_URL}/{station_id}/{date_str}/{hour:02d}"

    headers = {
        'DB-Client-Id': DB_CLIENT_ID,
        'DB-Api-Key': DB_API_KEY,
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    return response.text


def parse_timetable_xml(xml_str: str, target_lines: list = TARGET_LINES) -> list[dict]:
    """Parse timetable XML and extract target line arrivals.

    Args:
        xml_str: XML response string
        target_lines: List of line names to filter (e.g., ['RE5', 'RE6'])

    Returns:
        List of arrival records
    """
    root = ET.fromstring(xml_str)
    station_name = root.get('station', '')
    records = []

    for stop in root.findall('s'):
        stop_id = stop.get('id', '')

        # Get train line info
        tl = stop.find('tl')
        if tl is None:
            continue

        category = tl.get('c', '')  # RE, ICE, etc.
        train_number = tl.get('n', '')
        operator = tl.get('o', '')

        # Get arrival info
        ar = stop.find('ar')
        if ar is None:
            continue  # No arrival at this station

        line = ar.get('l', '')  # Line name (RE5, RE6, etc.)

        # Filter by target lines
        if line not in target_lines:
            continue

        # Parse arrival time (format: YYMMDDHHMM)
        pt = ar.get('pt', '')
        arrival_time = _parse_db_datetime(pt) if pt else None

        # Get platform and path
        platform = ar.get('pp', '')
        path = ar.get('ppth', '')  # Previous stops separated by |

        # Get departure info if exists
        dp = stop.find('dp')
        departure_time = None
        departure_platform = ''
        next_stops = ''

        if dp is not None:
            dp_pt = dp.get('pt', '')
            departure_time = _parse_db_datetime(dp_pt) if dp_pt else None
            departure_platform = dp.get('pp', '')
            next_stops = dp.get('ppth', '')

        record = {
            'fetch_time': datetime.now(),
            'station': station_name,
            'stop_id': stop_id,
            'line': line,
            'category': category,
            'train_number': train_number,
            'operator': operator,
            'arrival_time': arrival_time,
            'arrival_platform': platform,
            'previous_stops': path,
            'departure_time': departure_time,
            'departure_platform': departure_platform,
            'next_stops': next_stops,
        }
        records.append(record)

    return records


def _parse_db_datetime(dt_str: str) -> datetime:
    """Parse DB datetime format (YYMMDDHHMM) to datetime object."""
    if len(dt_str) != 10:
        return None

    try:
        year = 2000 + int(dt_str[0:2])
        month = int(dt_str[2:4])
        day = int(dt_str[4:6])
        hour = int(dt_str[6:8])
        minute = int(dt_str[8:10])
        return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError):
        return None


def save_to_daily_csv(records: list[dict], output_dir: Path,
                      fetch_time: datetime) -> Path:
    """Save records to daily partitioned CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    date_str = fetch_time.strftime('%Y%m%d')
    output_file = output_dir / f'db_timetable_{date_str}.csv'

    df = pd.DataFrame(records)

    if output_file.exists():
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

    return output_file


def save_latest_arrivals(records: list[dict], output_dir: Path,
                         max_arrivals: int = 3) -> Path | None:
    """Save latest arrivals in single-line format for frontend.

    Format: fetch_time,line1,time1,platform1,line2,time2,platform2,...
    Example: 2026-01-20 16:00,RE5,16:12,3,RE5,16:43,5,RE5,17:11,4
    """
    if not records:
        return None

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(records)

    # Filter future arrivals only
    now = datetime.now()
    df = df[df['arrival_time'] > now].copy()

    if df.empty:
        return None

    # Sort by arrival time and take top N
    df = df.sort_values('arrival_time').head(max_arrivals)

    # Build single-line format
    fetch_time_str = now.strftime('%Y-%m-%d %H:%M')
    parts = [fetch_time_str]

    for _, row in df.iterrows():
        line = row['line']
        arr_time = row['arrival_time'].strftime('%H:%M')
        platform = row['arrival_platform']
        parts.extend([line, arr_time, str(platform)])

    # Write as single line (append mode)
    latest_file = output_dir / 'latest_arrivals.csv'
    with open(latest_file, 'a') as f:
        f.write(','.join(parts) + '\n')

    return latest_file


# =============================================================================
# Main Task Function (called by Airflow)
# =============================================================================

def collect_gtfsrt_data(output_dir: Path = None) -> dict:
    """
    Main collection task: fetch, parse, and save timetable data.

    Returns:
        dict with collection statistics
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fetch_time = datetime.now()
    current_hour = fetch_time.hour

    print(f"Fetch time: {fetch_time}")
    print(f"Querying Berlin Hbf for hour {current_hour:02d}:00")

    # Fetch current hour and next hour for better coverage
    all_records = []

    for hour_offset in [0, 1]:
        hour = (current_hour + hour_offset) % 24
        try:
            xml_data = fetch_timetable(hour=hour)
            records = parse_timetable_xml(xml_data)
            all_records.extend(records)
            print(f"  Hour {hour:02d}: Found {len(records)} RE5/RE6 arrivals")
        except Exception as e:
            print(f"  Hour {hour:02d}: Error - {e}")

    # Deduplicate by stop_id
    seen = set()
    unique_records = []
    for r in all_records:
        if r['stop_id'] not in seen:
            seen.add(r['stop_id'])
            unique_records.append(r)

    print(f"Total unique RE5/RE6 arrivals: {len(unique_records)}")

    if not unique_records:
        print("No RE5/RE6 data found")
        return {'status': 'empty', 'records': 0}

    # Save daily CSV
    output_file = save_to_daily_csv(unique_records, output_dir, fetch_time)
    print(f"Saved {len(unique_records)} records to {output_file}")

    # Save latest arrivals
    latest_file = save_latest_arrivals(unique_records, output_dir)
    if latest_file:
        print(f"Saved latest arrivals to {latest_file}")

    return {
        'status': 'success',
        'records': len(unique_records),
        'output_file': str(output_file),
        'fetch_time': fetch_time.isoformat()
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    result = collect_gtfsrt_data()
    print(f"\nResult: {result}")
