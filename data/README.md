# Data Specifications

This document describes the data sources and schemas used in the transit seat availability prediction project.

## Directory Structure

```
data/
├── ingested/          # Raw data from external sources
├── processed/         # Cleaned and feature-engineered data
├── raw/               # Historical reference data
└── realtime/          # Real-time data collected by Airflow DAGs
    ├── weather/       # Weather forecasts
    └── gtfsrt/        # GTFS-RT transit data
```

---

## Real-time Weather Data

### Source
- **API**: [Open-Meteo](https://open-meteo.com/)
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **License**: Free for non-commercial use, CC BY 4.0
- **Update Frequency**: Every 5 minutes (via Airflow DAG)

### Data Type
- **Hourly Forecast**: 1-hour ahead prediction (not current conditions)
- **Location**: Berlin, Germany (52.52°N, 13.405°E)

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `fetch_time` | datetime | Time when data was fetched |
| `location` | string | Location name (Berlin) |
| `latitude` | float | Latitude coordinate |
| `longitude` | float | Longitude coordinate |
| `timezone` | string | Timezone (Europe/Berlin) |
| `forecast_time` | datetime | Target forecast time (+1 hour) |
| `forecast_hours_ahead` | int | Hours ahead of forecast (default: 1) |
| `temperature` | float | Temperature in °C |
| `apparent_temperature` | float | Feels-like temperature in °C |
| `humidity` | int | Relative humidity in % |
| `precipitation` | float | Precipitation in mm |
| `rain` | float | Rain in mm |
| `snowfall` | float | Snowfall in cm |
| `weather_code` | int | WMO weather code |
| `cloud_cover` | int | Cloud cover in % |
| `wind_speed` | float | Wind speed at 10m in km/h |
| `wind_direction` | int | Wind direction in degrees |
| `temperature_max` | float | Daily max temperature in °C |
| `temperature_min` | float | Daily min temperature in °C |
| `precipitation_sum` | float | Daily total precipitation in mm |
| `rain_sum` | float | Daily total rain in mm |
| `snowfall_sum` | float | Daily total snowfall in cm |
| `weather_description` | string | Human-readable weather description |

### WMO Weather Codes

| Code | Description |
|------|-------------|
| 0 | Clear sky |
| 1-3 | Mainly clear / Partly cloudy / Overcast |
| 45, 48 | Fog |
| 51, 53, 55 | Drizzle (light/moderate/dense) |
| 61, 63, 65 | Rain (slight/moderate/heavy) |
| 71, 73, 75 | Snow (slight/moderate/heavy) |
| 80-82 | Rain showers |
| 85, 86 | Snow showers |
| 95-99 | Thunderstorm |

### Output Files
- `weather_YYYYMMDD.csv`: Daily accumulated records
- `latest_weather.csv`: Most recent forecast snapshot

---

## Historical Weather Data (for Analysis)

### Source
- **API**: [Open-Meteo Archive](https://archive-api.open-meteo.com/)
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Coverage**: 2022-08-01 to 2024-06-30

### File
- `data/raw/berlin_weather_2022_2024.csv`

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `date` | date | Date |
| `temp_max` | float | Daily max temperature in °C |
| `temp_min` | float | Daily min temperature in °C |
| `temp_mean` | float | Daily mean temperature in °C |
| `precipitation` | float | Total precipitation in mm |
| `rain` | float | Total rain in mm |
| `snowfall` | float | Total snowfall in cm |
| `weather_code` | int | Dominant WMO weather code |

---

## Real-time Train Timetable Data

### Source
- **API**: [Deutsche Bahn Timetables API](https://developers.deutschebahn.com/)
- **Endpoint**: `https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1/plan/{station}/{date}/{hour}`
- **Format**: XML
- **Authentication**: API Key required (DB-Client-Id, DB-Api-Key headers)
- **Update Frequency**: Every 5 minutes (via Airflow DAG)

### Station
- **Berlin Hauptbahnhof (Berlin Hbf)**
- **EVA Number**: 8098160

### Filtered Lines
- RE5 (Regional Express)
- RE6 (Regional Express)

### API Request Format
```
GET /plan/{eva_number}/{YYMMDD}/{HH}
Headers:
  DB-Client-Id: <client_id>
  DB-Api-Key: <api_key>
```

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `fetch_time` | datetime | Time when data was fetched |
| `station` | string | Station name |
| `stop_id` | string | Unique stop identifier |
| `line` | string | Line name (RE5, RE6) |
| `category` | string | Train category (RE) |
| `train_number` | string | Train number |
| `operator` | string | Operator code |
| `arrival_time` | datetime | Scheduled arrival time |
| `arrival_platform` | string | Arrival platform |
| `previous_stops` | string | Previous stops (pipe-separated) |
| `departure_time` | datetime | Scheduled departure time |
| `departure_platform` | string | Departure platform |
| `next_stops` | string | Next stops (pipe-separated) |
| `minutes_until_arrival` | float | Minutes until arrival (in latest_arrivals.csv) |

### Output Files
- `db_timetable_YYYYMMDD.csv`: Daily accumulated records
- `latest_arrivals.csv`: Upcoming arrivals snapshot

---

## RE5/RE6 Occupancy Data

### Source
- Deutsche Bahn occupancy data (historical)
- Coverage: 2022-08 to 2024-06

### Processed Files
- `data/processed/re5_processed.csv`
- `data/processed/re6_processed.csv`

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `trip_start_date` | datetime | Trip date |
| `trip_dep_time` | datetime | Departure time |
| `trip_arr_time` | datetime | Arrival time |
| `occupancy` | int | Passenger count |
| `capacity` | int | Seat capacity |
| `hour` | int | Hour of departure (0-23) |
| `dow` | int | Day of week (0=Mon, 6=Sun) |
| `is_rush_hour` | bool | Rush hour flag (6-9, 16-19) |
| `is_weekend` | bool | Weekend flag |
| `occ_pct` | float | Occupancy percentage |
