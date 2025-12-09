import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

# Train and forecast
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
    forecast_scaled = results.forecast(steps=horizon, exog=discount_plan['Discount']) # type: ignore
    forecast_values = scaler_y.inverse_transform(forecast_scaled.values.reshape(-1, 1))
    forecast_final = np.round(np.maximum(forecast_values, 0)).flatten().tolist()
    
    return forecast_final, horizon