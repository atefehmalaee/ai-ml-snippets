"""
Facebook Prophet template.
pip install prophet
"""
import pandas as pd
from prophet import Prophet

def fit_prophet(df: pd.DataFrame):
    # df must have columns: ds (datetime), y (value)
    m = Prophet()
    m.fit(df)
    return m

def forecast_future(model: Prophet, periods=30, freq="D"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
