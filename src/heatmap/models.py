from datetime import date
from typing import Optional
from sqlmodel import SQLModel, Field

class HeatMap(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    address: str = Field(index=True)
    latitude: float
    longitude: float
    product: str = Field(index=True)
    daily_demand: float
    target_date: date = Field(index=True)
    last_updated: date = Field(default_factory=date.today)