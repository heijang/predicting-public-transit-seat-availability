# Transit Information API

FastAPI backend for transit information service.

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.main:app --reload
```

## API Documentation

Interactive docs available at: http://localhost:8000/docs

### Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

#### POST /transit

Get transit information between stations.

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| departure_station | string | Departure station name |
| arrival_station | string | Arrival station name |
| departure_time | string | Departure time (HH:MM format) |

**Example Request:**
```json
{
  "departure_station": "Seoul",
  "arrival_station": "Busan",
  "departure_time": "09:00"
}
```

**Response:**
| Field | Type | Description |
|-------|------|-------------|
| total_minutes | int | Total travel time in minutes |
| stops | int | Number of stops |
| summary | string | Summary text (e.g., "~45min (5 stops)") |
| next_trains | array | Next trains list (max 3) |
| todays_occupancy | array | 24 hourly occupancy values (0.0-1.0) for line graph |
| recommendation | string | Recommendation message |

**next_trains item:**
| Field | Type | Description |
|-------|------|-------------|
| arrival_minutes | int | Minutes until arrival |
| congestion_value | float | Congestion value (0.0-1.0) |
| congestion_code | int | Congestion code (1-10) |
| congestion_level | string | "High", "Normal", or "Low" |

**Example Response:**
```json
{
  "total_minutes": 45,
  "stops": 5,
  "summary": "~45min (5 stops)",
  "next_trains": [
    {
      "arrival_minutes": 3,
      "congestion_value": 0.8,
      "congestion_code": 8,
      "congestion_level": "High"
    },
    {
      "arrival_minutes": 8,
      "congestion_value": 0.5,
      "congestion_code": 5,
      "congestion_level": "Normal"
    },
    {
      "arrival_minutes": 15,
      "congestion_value": 0.2,
      "congestion_code": 2,
      "congestion_level": "Low"
    }
  ],
  "todays_occupancy": [0.1, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1],
  "recommendation": "The next train is crowded. Consider taking the train arriving in 8 minutes for a more comfortable ride."
}
```
