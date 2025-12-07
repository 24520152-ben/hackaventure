import io
import uvicorn
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

app = FastAPI(title='Demand Forecasting API', description='HACKAVENTURE')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train_and_forecast(sales: pd.DataFrame, discount_plan: pd.DataFrame, product: str):
    # Sale processing
    required_columns = ['Date', 'Product', 'Quantity', 'Discount']
    if not all(col in sales.columns for col in required_columns):
        raise ValueError(f'CSV file has missing column. Required columns: {required_columns}.')

    sales = sales[sales['Product'] == product]
    if sales.empty:
        raise ValueError(f'No data found for product {product} in CSV file.')

    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values(by='Date', ascending=True)
    sales = sales.set_index('Date')

    sales = sales.resample('D').agg({
        'Quantity': 'sum',
        'Discount': 'mean',
    })
    sales['Quantity'] = sales['Quantity'].fillna(0)
    sales['Discount'] = sales['Discount'].fillna(0)
    if len(sales) < 14:
        raise ValueError(f'Not enough data to forecast (a minimum of 14 days is required).')

    # Discount plan processing
    if 'Discount' not in discount_plan.columns:
        raise ValueError(f'CSV file has missing column. Required column: Discount.')

    discount_plan['Discount'] = discount_plan['Discount'].fillna(0)

    horizon = len(discount_plan)
    if horizon == 0:
        raise ValueError(f'Empty discount plan.')

    # Scaling
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    sales['Quantity'] = scaler_y.fit_transform(sales[['Quantity']])
    sales['Discount'] = scaler_X.fit_transform(sales[['Discount']])
    discount_plan['Discount'] = scaler_X.transform(discount_plan[['Discount']])

    # Model training
    model = SARIMAX(
        endog=sales['Quantity'],
        exog=sales['Discount'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
    )
    results = model.fit()

    # Forecasting
    forecast_scaled = results.forecast(steps=horizon, exog=discount_plan['Discount'])
    forecast_values = scaler_y.inverse_transform(forecast_scaled.values.reshape(-1, 1))
    forecast_final = np.round(np.maximum(forecast_values, 0)).flatten().tolist()
    
    return forecast_final, horizon

@app.get('/')
def home():
    return {'message': 'Demand Forecasting API is running'}

@app.post('/forecast')
async def forecast_demand(
    address: str = Form(...),
    product: str = Form(...),
    sales_csv: UploadFile = File(..., description=f'CSV file with these columns: [Date, Quantity, Discount] in the past. At least 14 days to forecast.'),
    discount_plan_csv: UploadFile = File(..., description=f'CSV file with a column: [Discount] in the future. If you want to forecast the next 7 days, Discount must have 7 rows.'),
):
    try:
        sales = await sales_csv.read()
        sales = pd.read_csv(io.BytesIO(sales))
        discount_plan = await discount_plan_csv.read()
        discount_plan = pd.read_csv(io.BytesIO(discount_plan))

        forecast, horizon = train_and_forecast(sales, discount_plan, product)
        total_demand = sum(forecast)

        return {
            'Store Address': address,
            'Product': product,
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

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)