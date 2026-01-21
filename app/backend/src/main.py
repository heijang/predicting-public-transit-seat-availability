import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models import CongestionLevel, TrainInfo, TransitRequest, TransitResponse

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
LATEST_ARRIVALS_PATH = DATA_DIR / "realtime" / "gtfsrt" / "latest_arrivals.csv"

app = FastAPI(title="Transit Information API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTLCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 60):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def _make_key(self, request: TransitRequest) -> str:
        return f"{request.departure_station}:{request.arrival_station}:{request.departure_time}"

    def get(self, request: TransitRequest) -> TransitResponse | None:
        key = self._make_key(request)
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, request: TransitRequest, response: TransitResponse) -> None:
        key = self._make_key(request)
        self._cache[key] = (time.time(), response)


transit_cache = TTLCache(ttl_seconds=60)


def get_congestion_level(value: float) -> CongestionLevel:
    """Convert congestion value to level."""
    if value >= 0.7:
        return CongestionLevel.HIGH
    elif value >= 0.4:
        return CongestionLevel.NORMAL
    return CongestionLevel.LOW


def get_dummy_congestion() -> tuple[float, int, CongestionLevel]:
    """Generate dummy congestion data. To be replaced with model prediction."""
    import random
    value = round(random.uniform(0.2, 0.9), 2)
    code = max(1, min(10, int(value * 10)))
    level = get_congestion_level(value)
    return value, code, level


def parse_latest_arrivals(departure_time: str) -> list[TrainInfo]:
    """Parse latest arrivals from CSV and calculate minutes until arrival."""
    if not LATEST_ARRIVALS_PATH.exists():
        return []

    with open(LATEST_ARRIVALS_PATH) as f:
        first_line = f.readline().strip()

    if not first_line:
        return []

    parts = first_line.split(",")
    if len(parts) < 4:
        return []

    # Parse current timestamp from CSV (format: "2026-01-20 16:10")
    csv_timestamp = parts[0]
    csv_time = csv_timestamp.split(" ")[1]  # Get "16:10" part
    csv_hour, csv_min = map(int, csv_time.split(":"))
    current_total_minutes = csv_hour * 60 + csv_min

    trains: list[TrainInfo] = []
    # Skip first element (current timestamp), then process in groups of 3: [train, arrival_time, platform]
    i = 1
    while i + 2 < len(parts) and len(trains) < 3:
        train_line = parts[i]
        arrival_time = parts[i + 1]
        platform = int(parts[i + 2])

        # Parse arrival time (HH:MM format)
        arr_hour, arr_min = map(int, arrival_time.split(":"))
        arr_total_minutes = arr_hour * 60 + arr_min

        # Calculate minutes until arrival from current time in CSV
        arrival_minutes = arr_total_minutes - current_total_minutes
        if arrival_minutes < 0:
            arrival_minutes += 24 * 60  # Handle day wrap

        # Get congestion (dummy for now)
        congestion_value, congestion_code, congestion_level = get_dummy_congestion()

        trains.append(TrainInfo(
            train_line=train_line,
            arrival_minutes=arrival_minutes,
            platform=platform,
            congestion_value=congestion_value,
            congestion_code=congestion_code,
            congestion_level=congestion_level,
        ))

        i += 3

    return trains


def get_dummy_trains() -> list[TrainInfo]:
    """Generate dummy next trains data (fallback)."""
    return [
        TrainInfo(
            train_line="RE5",
            arrival_minutes=3,
            platform=3,
            congestion_value=0.8,
            congestion_code=8,
            congestion_level=CongestionLevel.HIGH,
        ),
        TrainInfo(
            train_line="RE5",
            arrival_minutes=8,
            platform=5,
            congestion_value=0.5,
            congestion_code=5,
            congestion_level=CongestionLevel.NORMAL,
        ),
        TrainInfo(
            train_line="RE5",
            arrival_minutes=15,
            platform=4,
            congestion_value=0.2,
            congestion_code=2,
            congestion_level=CongestionLevel.LOW,
        ),
    ]


def get_dummy_occupancy() -> list[float]:
    """Generate dummy hourly occupancy data for 24 hours."""
    return [
        0.1, 0.1, 0.05, 0.05, 0.1, 0.3,  # 00:00 - 05:00
        0.5, 0.8, 0.9, 0.7, 0.5, 0.4,    # 06:00 - 11:00
        0.6, 0.5, 0.4, 0.5, 0.6, 0.8,    # 12:00 - 17:00
        0.9, 0.7, 0.5, 0.3, 0.2, 0.1,    # 18:00 - 23:00
    ]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/transit", response_model=TransitResponse)
async def get_transit_info(request: TransitRequest) -> TransitResponse:
    """Get transit information from realtime data. Responses are cached for 1 minute."""
    cached = transit_cache.get(request)
    if cached is not None:
        return cached

    total_minutes = 45
    stops = 5

    # Try to get realtime data, fallback to dummy
    next_trains = parse_latest_arrivals(request.departure_time)
    if not next_trains:
        next_trains = get_dummy_trains()

    first_train = next_trains[0]

    if first_train.congestion_level == CongestionLevel.HIGH:
        recommendation = f"The next train is crowded. Consider taking the train arriving in {next_trains[1].arrival_minutes} minutes for a more comfortable ride."
    elif first_train.congestion_level == CongestionLevel.LOW:
        recommendation = "The next train has plenty of space. Good time to travel!"
    else:
        recommendation = "Normal congestion levels. Have a good trip!"

    response = TransitResponse(
        total_minutes=total_minutes,
        stops=stops,
        summary=f"~{total_minutes}min ({stops} stops)",
        next_trains=next_trains,
        todays_occupancy=get_dummy_occupancy(),
        recommendation=recommendation,
    )

    transit_cache.set(request, response)
    return response
