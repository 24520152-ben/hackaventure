import io
import pandas as pd
from sqlmodel import Session, select
from datetime import date, timedelta
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from src.config import logger
from src.database import get_session
from src.heatmap.models import HeatMap
from src.forecast.service import train_and_forecast

router = APIRouter(tags=['Forecast'])

@router.post('/forecast')
def forecast_demand(
    address: str = Form(...),
    product: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    sales_csv: UploadFile = File(..., description=f'CSV file with these columns: [Date, Quantity, Discount] in the past. At least 14 days to forecast.'),
    discount_plan_csv: UploadFile = File(..., description=f'CSV file with a column: [Discount] in the future. If you want to forecast the next 7 days, Discount must have 7 rows.'),
    db: Session = Depends(get_session),
):
    try:
        # Read CSV
        sales = sales_csv.file.read()
        sales = pd.read_csv(io.BytesIO(sales))
        discount_plan = discount_plan_csv.file.read()
        discount_plan = pd.read_csv(io.BytesIO(discount_plan))

        # Forecast
        forecast, horizon = train_and_forecast(sales, discount_plan, product)
        total_demand = sum(forecast)

        today = date.today()
        for i, daily_demand in enumerate(forecast, start=1):
            # Start forecasting from tomorrow
            current_target_date = today + timedelta(days=i)

            statement = select(HeatMap).where(
                HeatMap.address == address,
                HeatMap.product == product,
                HeatMap.latitude == latitude,
                HeatMap.longitude == longitude,
                HeatMap.target_date == current_target_date,
            )
            existing_entry = db.exec(statement).first()

            if existing_entry:
                # Update existing entry
                existing_entry.daily_demand = daily_demand
                existing_entry.last_updated = date.today()
                db.add(existing_entry)
            else:
                # Insert new entry
                new_entry = HeatMap(
                    address=address,
                    product=product,
                    latitude=latitude,
                    longitude=longitude,
                    daily_demand=daily_demand,
                    target_date=current_target_date,
                )
                db.add(new_entry)
        
        db.commit()

        return {
            'Store Address': address,
            'Product': product,
            'Latitude': latitude,
            'Longitude': longitude,
            'Forecast Horizon': horizon,
            'Daily Forecast': forecast,
            'Total Demand': total_demand,
        }

    except ValueError as ve:
        logger.warning(f'User Input Error: {str(ve)}')
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception('Internal Server Error')
        raise HTTPException(status_code=500, detail=str(e))