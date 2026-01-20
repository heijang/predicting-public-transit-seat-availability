# Airflow Pipelines

This directory contains Apache Airflow configuration and logs for the Transit Seat Availability project.

## DAGs Overview

| DAG ID | Schedule | Description |
|--------|----------|-------------|
| `gtfsrt_collector` | Every 5 minutes | Real-time GTFS-RT and weather data collection |
| `transit_seat_availability_pipeline` | Manual | ML pipeline for seat availability prediction |

---

## 1. GTFS-RT Collector DAG

**DAG ID:** `gtfsrt_collector`
**Schedule:** `*/5 * * * *` (every 5 minutes)
**Tags:** `gtfsrt`, `realtime`, `transit`

### Description

Collects real-time data from multiple sources in parallel for RE5/RE6 train lines in Berlin.

### Task Flow

```
start >> [collect_gtfsrt, collect_weather] >> end
```

### Tasks

#### collect_gtfsrt

Fetches GTFS-RT (General Transit Feed Specification - Realtime) data from VBB.

- **Source:** `https://production.gtfsrt.vbb.de/data`
- **Filter:** RE5, RE6 train lines only
- **Output Directory:** `data/realtime/gtfsrt/`

**Output Files:**

| File | Description |
|------|-------------|
| `gtfsrt_YYYYMMDD.csv` | Daily partitioned trip updates (appended every 5 min) |
| `latest_arrivals.csv` | Next 3 trains per stop (overwritten each run) |

**Fields Collected:**

| Field | Description |
|-------|-------------|
| `fetch_time` | Timestamp of data fetch |
| `route_id` | Train line identifier |
| `trip_id` | Unique trip identifier |
| `direction` | Direction of travel (0/1) |
| `stop_id` | Stop identifier |
| `stop_sequence` | Order of stop in trip |
| `arrival_time` | Predicted arrival time |
| `arrival_delay_sec` | Delay in seconds |
| `departure_time` | Predicted departure time |
| `departure_delay_sec` | Departure delay in seconds |
| `minutes_until_arrival` | Minutes until train arrives |

#### collect_weather

Fetches real-time weather data for Berlin from Open-Meteo API.

- **Source:** Open-Meteo API (free, no API key required)
- **Location:** Berlin (52.52, 13.405)
- **Output Directory:** `data/realtime/weather/`

**Output Files:**

| File | Description |
|------|-------------|
| `weather_YYYYMMDD.csv` | Daily partitioned weather data |
| `latest_weather.csv` | Current weather snapshot |

**Fields Collected:**

| Field | Description |
|-------|-------------|
| `temperature` | Current temperature (°C) |
| `apparent_temperature` | Feels-like temperature (°C) |
| `temperature_max` | Daily maximum temperature (°C) |
| `temperature_min` | Daily minimum temperature (°C) |
| `humidity` | Relative humidity (%) |
| `precipitation` | Current precipitation (mm) |
| `precipitation_sum` | Daily precipitation total (mm) |
| `rain` | Current rain (mm) |
| `snowfall` | Current snowfall (cm) |
| `weather_code` | WMO weather code |
| `weather_description` | Human-readable weather |
| `cloud_cover` | Cloud cover (%) |
| `wind_speed` | Wind speed at 10m (km/h) |
| `wind_direction` | Wind direction (degrees) |

### Source Code

- DAG: `dags/gtfsrt_collector_dag.py`
- GTFS-RT Logic: `src/gtfsrt_collector.py`
- Weather Logic: `src/weather_collector.py`

---

## 2. Transit Seat Availability Pipeline DAG

**DAG ID:** `transit_seat_availability_pipeline`
**Schedule:** Manual (triggered on-demand)
**Tags:** `transit`, `machine-learning`

### Description

End-to-end machine learning pipeline for predicting seat availability on RE5/RE6 trains.

### Task Flow

```
load_data >> preprocess >> feature_engineer >> train_model >> evaluate
```

### Tasks

| Task | Description | Source |
|------|-------------|--------|
| `load_data` | Load raw occupancy data from CSV | `src/data_loader.py` |
| `preprocess` | Clean and standardize data, create time features | `src/preprocess.py` |
| `feature_engineer` | Create ML features with target variables | `src/feature_engineer.py` |
| `train_model` | Train Random Forest models for multiple horizons | `src/model_trainer.py` |
| `evaluate` | Evaluate model performance | `src/evaluate.py` |

### Prediction Horizons

The pipeline trains models for multiple prediction horizons:

| Horizon | Description |
|---------|-------------|
| 30 min | Predict occupancy 30 minutes ahead |
| 1 hour | Predict occupancy 1 hour ahead |
| 3 hours | Predict occupancy 3 hours ahead |
| Next day | Predict occupancy for the next day |

### Output

- Models: `outputs/rq1_models/`
- Results: `outputs/rq1_results/`

---

## Directory Structure

```
airflow/
├── config/           # Airflow configuration files
├── logs/             # Task execution logs
├── outputs/          # Pipeline outputs
└── README.md         # This file

dags/
├── gtfsrt_collector_dag.py              # Real-time data collection DAG
└── transit_seat_availability_pipeline_dag.py  # ML pipeline DAG

src/
├── gtfsrt_collector.py    # GTFS-RT collection logic
├── weather_collector.py   # Weather collection logic
├── data_loader.py         # Data loading utilities
├── preprocess.py          # Preprocessing functions
├── feature_engineer.py    # Feature engineering
├── model_trainer.py       # Model training
└── evaluate.py            # Model evaluation
```

---

## Running the Pipelines

### Start Airflow

```bash
docker-compose up -d
```

Access Airflow UI at: http://localhost:8080

### Trigger DAGs

**Via UI:**
1. Navigate to DAGs page
2. Toggle DAG to enable
3. Click "Trigger DAG" button

**Via CLI:**
```bash
# Trigger ML pipeline
docker exec transit-seat-availability-pipeline-airflow airflow dags trigger transit_seat_availability_pipeline

# Trigger real-time collector (runs automatically every 5 min)
docker exec transit-seat-availability-pipeline-airflow airflow dags trigger gtfsrt_collector
```

---

## Configuration

Default arguments for all DAGs:

| Parameter | Value |
|-----------|-------|
| `owner` | transit-team |
| `depends_on_past` | False |
| `start_date` | 2026-01-01 |
| `email_on_failure` | False |
| `retries` | 1-2 |
| `retry_delay` | 30s - 5min |
