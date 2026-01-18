from pydantic import BaseModel, Field
from enum import Enum


class CongestionLevel(str, Enum):
    HIGH = "High"
    NORMAL = "Normal"
    LOW = "Low"


class TrainInfo(BaseModel):
    arrival_minutes: int = Field(..., description="Minutes until arrival")
    congestion_value: float = Field(..., ge=0.0, le=1.0, description="Congestion value from 0.0 to 1.0")
    congestion_code: int = Field(..., ge=1, le=10, description="Congestion code from 1 to 10")
    congestion_level: CongestionLevel = Field(..., description="Congestion text: High, Normal, or Low")


class TransitRequest(BaseModel):
    departure_station: str
    arrival_station: str
    departure_time: str = Field(..., description="Departure time in HH:MM format")


class TransitResponse(BaseModel):
    total_minutes: int = Field(..., description="Total travel time in minutes")
    stops: int = Field(..., description="Number of stops")
    summary: str = Field(..., description="e.g., '~60min (5 stops)'")
    next_trains: list[TrainInfo] = Field(..., max_length=3, description="Next trains list, max 3")
    todays_occupancy: list[float] = Field(..., description="Hourly occupancy data for line graph (24 values)")
    recommendation: str = Field(..., description="Recommendation message")
