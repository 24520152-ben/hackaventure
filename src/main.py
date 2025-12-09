import os
import uvicorn
from datetime import date
from fastapi import FastAPI
from sqlmodel import Session, delete
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger
from src.heatmap.models import HeatMap
from src.heatmap.router import router as heatmap_router
from src.forecast.router import router as forecast_router
from src.database import create_database_and_table, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_database_and_table()

    # Delete entries have target date <= today
    with Session(engine) as session:
        statement = delete(HeatMap).where(HeatMap.target_date <= date.today()) # type: ignore
        session.exec(statement)
        session.commit()
    
    yield

# FastAPI configuration
app = FastAPI(title='Demand Forecasting API', description='HACKAVENTURE', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://hackaventure-fe.vercel.app'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(heatmap_router)
app.include_router(forecast_router)

@app.get('/')
def home():
    return {'message': 'Demand Forecasting API is running'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)