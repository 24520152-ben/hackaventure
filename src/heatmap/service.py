from sqlmodel import Session, delete
from datetime import date
from src.heatmap.models import HeatMap

def delete_expired_entries(session: Session):
    statement = delete(HeatMap).where(HeatMap.target_date <= date.today()) # type: ignore
    session.exec(statement)
    session.commit()