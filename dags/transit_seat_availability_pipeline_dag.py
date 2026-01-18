"""Public Transit Seat Availability Pipeline DAG"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Default Arguments
# =============================================================================

default_args = {
    'owner': 'transit-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# =============================================================================
# Task Functions
# =============================================================================

def load_data(**context):
    """Load raw data"""
    from src.data_loader import load_occupancies

    df = load_occupancies()
    print(f"Loaded {len(df)} records")


def preprocess(**context):
    """Preprocess data"""
    from src.preprocess import main as preprocess_main

    preprocess_main()


def feature_engineer(**context):
    """Feature engineering"""
    from src.feature_engineer import main as feature_engineer_main

    feature_engineer_main()


def train_model(**context):
    """Train model"""
    from src.model_trainer import train_all_horizons

    results = train_all_horizons()
    print(results)


def evaluate(**context):
    """Evaluate model"""
    from src.evaluate import main as evaluate_main

    evaluate_main()


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    'transit_seat_availability_pipeline',
    default_args=default_args,
    description='Transit seat availability prediction pipeline',
    schedule=None,
    catchup=False,
    tags=['transit', 'machine-learning'],
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess,
    )

    feature_engineer_task = PythonOperator(
        task_id='feature_engineer',
        python_callable=feature_engineer,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate,
    )

    # Task dependencies
    load_data_task >> preprocess_task >> feature_engineer_task >> train_model_task >> evaluate_task
