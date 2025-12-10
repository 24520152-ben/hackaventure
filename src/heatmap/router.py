from sqlmodel import Session, select
from datetime import date, timedelta
from fastapi import APIRouter, Depends, BackgroundTasks

from src.database import get_session
from src.heatmap.models import HeatMap
from src.heatmap.service import delete_expired_entries

router = APIRouter(tags=['Heatmap'])

@router.get('/products')
def get_unique_products(db: Session = Depends(get_session)):
    statement = select(HeatMap.product).distinct()
    results = db.exec(statement).all()
    return {
        'products': results,
    }

@router.get('/heatmap')
def get_heatmap_by_product(
    product: str,
    background_tasks: BackgroundTasks,
    horizon: int = 7,
    db: Session = Depends(get_session),
):
    background_tasks.add_task(delete_expired_entries, db)

    today = date.today()
    last_day = today + timedelta(days=horizon)
    statement = select(HeatMap).where(HeatMap.product == product, HeatMap.target_date > today, HeatMap.target_date <= last_day)
    results = db.exec(statement).all()
    return {
        'product': product,
        'horizon': horizon,
        'data': results,
    }