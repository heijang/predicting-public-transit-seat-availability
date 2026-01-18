from fastapi import FastAPI

from src.models import CongestionLevel, TrainInfo, TransitRequest, TransitResponse

app = FastAPI(title="Transit Information API", version="0.1.0")


def get_dummy_trains() -> list[TrainInfo]:
    """Generate dummy next trains data."""
    return [
        TrainInfo(
            arrival_minutes=3,
            congestion_value=0.8,
            congestion_code=8,
            congestion_level=CongestionLevel.HIGH,
        ),
        TrainInfo(
            arrival_minutes=8,
            congestion_value=0.5,
            congestion_code=5,
            congestion_level=CongestionLevel.NORMAL,
        ),
        TrainInfo(
            arrival_minutes=15,
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
    """Get transit information with dummy data."""
    total_minutes = 45
    stops = 5

    next_trains = get_dummy_trains()
    first_train = next_trains[0]

    if first_train.congestion_level == CongestionLevel.HIGH:
        recommendation = f"The next train is crowded. Consider taking the train arriving in {next_trains[1].arrival_minutes} minutes for a more comfortable ride."
    elif first_train.congestion_level == CongestionLevel.LOW:
        recommendation = "The next train has plenty of space. Good time to travel!"
    else:
        recommendation = "Normal congestion levels. Have a good trip!"

    return TransitResponse(
        total_minutes=total_minutes,
        stops=stops,
        summary=f"~{total_minutes}min ({stops} stops)",
        next_trains=next_trains,
        todays_occupancy=get_dummy_occupancy(),
        recommendation=recommendation,
    )
