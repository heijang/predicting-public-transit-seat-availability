"""GTFS-RT Real-time Feed Collector

Collects RE5, RE6 train arrival information from VBB GTFS-RT feed.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from google.transit import gtfs_realtime_pb2


# =============================================================================
# Configuration
# =============================================================================

GTFSRT_URL = 'https://production.gtfsrt.vbb.de/data'
TARGET_LINES = ['RE5', 'RE6']
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'realtime' / 'gtfsrt'


# =============================================================================
# Core Functions
# =============================================================================

def fetch_gtfsrt_feed(url: str = GTFSRT_URL, timeout: int = 30) -> gtfs_realtime_pb2.FeedMessage:
    """Fetch and parse GTFS-RT feed from URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)

    return feed


def extract_trip_updates(feed: gtfs_realtime_pb2.FeedMessage,
                         target_lines: list = TARGET_LINES) -> list[dict]:
    """Extract trip updates for target lines from GTFS-RT feed."""
    feed_timestamp = datetime.fromtimestamp(feed.header.timestamp)
    records = []

    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue

        tu = entity.trip_update
        route_id = tu.trip.route_id

        # Filter by target lines
        is_target = any(line in route_id.upper() for line in target_lines)
        if not is_target:
            is_target = any(line in tu.trip.trip_id.upper() for line in target_lines)

        if not is_target:
            continue

        # Process each stop update
        for stu in tu.stop_time_update:
            record = _parse_stop_time_update(
                stu, tu, route_id, feed_timestamp
            )
            records.append(record)

    return records


def _parse_stop_time_update(stu, trip_update, route_id: str,
                            feed_timestamp: datetime) -> dict:
    """Parse a single stop time update into a record dict."""
    # Arrival info
    arrival_time = None
    arrival_delay = 0
    if stu.HasField('arrival'):
        arrival_delay = stu.arrival.delay
        if stu.arrival.time > 0:
            arrival_time = datetime.fromtimestamp(stu.arrival.time)

    # Departure info
    departure_time = None
    departure_delay = 0
    if stu.HasField('departure'):
        departure_delay = stu.departure.delay
        if stu.departure.time > 0:
            departure_time = datetime.fromtimestamp(stu.departure.time)

    return {
        'fetch_time': feed_timestamp,
        'route_id': route_id,
        'trip_id': trip_update.trip.trip_id,
        'direction': trip_update.trip.direction_id,
        'start_date': trip_update.trip.start_date,
        'start_time': trip_update.trip.start_time,
        'stop_id': stu.stop_id,
        'stop_sequence': stu.stop_sequence,
        'arrival_time': arrival_time,
        'arrival_delay_sec': arrival_delay,
        'departure_time': departure_time,
        'departure_delay_sec': departure_delay,
    }


def create_dataframe(records: list[dict], feed_timestamp: datetime) -> pd.DataFrame:
    """Create DataFrame from records with computed fields."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Calculate minutes until arrival
    df['minutes_until_arrival'] = None
    if 'arrival_time' in df.columns:
        mask = df['arrival_time'].notna()
        df.loc[mask, 'minutes_until_arrival'] = (
            (df.loc[mask, 'arrival_time'] - feed_timestamp).dt.total_seconds() / 60
        ).round(1)

    return df


def save_to_daily_csv(df: pd.DataFrame, output_dir: Path,
                      feed_timestamp: datetime) -> Path:
    """Save DataFrame to daily partitioned CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    date_str = feed_timestamp.strftime('%Y%m%d')
    output_file = output_dir / f'gtfsrt_{date_str}.csv'

    if output_file.exists():
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

    return output_file


def save_latest_summary(df: pd.DataFrame, output_dir: Path,
                        n_trains: int = 3) -> Path | None:
    """Save summary with next N trains per stop."""
    if df.empty:
        return None

    # Filter future arrivals only
    future = df[df['minutes_until_arrival'] > 0].copy()

    if future.empty:
        return None

    # Sort by stop and arrival time, get next N trains per stop
    future = future.sort_values(['stop_id', 'arrival_time'])
    summary = future.groupby('stop_id').head(n_trains)

    # Select key columns
    summary = summary[[
        'fetch_time', 'stop_id', 'route_id', 'trip_id',
        'arrival_time', 'minutes_until_arrival', 'arrival_delay_sec'
    ]].copy()

    # Save summary
    summary_file = output_dir / 'latest_arrivals.csv'
    summary.to_csv(summary_file, index=False)

    return summary_file


# =============================================================================
# Main Task Functions (called by Airflow)
# =============================================================================

def collect_gtfsrt_data(output_dir: Path = None) -> dict:
    """
    Main collection task: fetch, parse, and save GTFS-RT data.

    Returns:
        dict with collection statistics
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Fetch feed
    feed = fetch_gtfsrt_feed()
    feed_timestamp = datetime.fromtimestamp(feed.header.timestamp)

    print(f"Feed timestamp: {feed_timestamp}")
    print(f"Total entities: {len(feed.entity)}")

    # Extract records
    records = extract_trip_updates(feed)
    print(f"Found {len(records)} RE5/RE6 stop updates")

    if not records:
        print("No RE5/RE6 data found in this fetch")
        return {'status': 'empty', 'records': 0}

    # Create DataFrame
    df = create_dataframe(records, feed_timestamp)

    # Save daily CSV
    output_file = save_to_daily_csv(df, output_dir, feed_timestamp)
    print(f"Saved {len(df)} records to {output_file}")

    # Save latest summary
    summary_file = save_latest_summary(df, output_dir)
    if summary_file:
        print(f"Saved summary to {summary_file}")

    return {
        'status': 'success',
        'records': len(df),
        'output_file': str(output_file),
        'feed_timestamp': feed_timestamp.isoformat()
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    result = collect_gtfsrt_data()
    print(f"Result: {result}")
