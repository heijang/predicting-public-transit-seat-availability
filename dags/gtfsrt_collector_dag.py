"""GTFS-RT Real-time Feed Collector DAG

Collects RE5, RE6 train arrival information every 5 minutes.
Structured for future parallel task expansion.
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collectors import collect_gtfsrt_data, collect_weather_data


# =============================================================================
# Configuration
# =============================================================================

GTFSRT_OUTPUT_DIR = project_root / 'data' / 'realtime' / 'gtfsrt'
WEATHER_OUTPUT_DIR = project_root / 'data' / 'realtime' / 'weather'


# =============================================================================
# Default Arguments
# =============================================================================

default_args = {
    'owner': 'transit-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}


# =============================================================================
# Task Wrappers
# =============================================================================

def task_collect_gtfsrt(**context):
    """Wrapper for GTFS-RT collection task."""
    result = collect_gtfsrt_data(output_dir=GTFSRT_OUTPUT_DIR)
    return result


def task_collect_weather(**context):
    """Wrapper for weather collection task."""
    result = collect_weather_data(output_dir=WEATHER_OUTPUT_DIR)
    return result


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    'gtfsrt_collector',
    default_args=default_args,
    description='Collect RE5/RE6 GTFS-RT data every 5 minutes',
    schedule='*/5 * * * *',  # Every 5 minutes
    catchup=False,
    max_active_runs=1,
    tags=['gtfsrt', 'realtime', 'transit'],
) as dag:

    # Start task
    start = EmptyOperator(task_id='start')

    # ==========================================================================
    # Parallel Task Group: Data Collection
    # Add more collection tasks here for parallel execution
    # ==========================================================================

    collect_gtfsrt = PythonOperator(
        task_id='collect_gtfsrt',
        python_callable=task_collect_gtfsrt,
        execution_timeout=timedelta(seconds=45),
    )

    collect_weather = PythonOperator(
        task_id='collect_weather',
        python_callable=task_collect_weather,
        execution_timeout=timedelta(seconds=30),
    )

    # Example: Future parallel tasks can be added here
    # collect_events = PythonOperator(
    #     task_id='collect_events',
    #     python_callable=task_collect_events,
    #     execution_timeout=timedelta(seconds=30),
    # )

    # ==========================================================================
    # End task (join point for parallel tasks)
    # ==========================================================================

    end = EmptyOperator(task_id='end')

    # ==========================================================================
    # Task Dependencies
    # ==========================================================================

    # Parallel execution: start >> [collect_gtfsrt, collect_weather] >> end

    start >> [collect_gtfsrt, collect_weather] >> end
