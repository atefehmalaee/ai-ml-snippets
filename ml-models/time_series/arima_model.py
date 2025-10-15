"""
ARIMA / SARIMAX quick template.
pip install statsmodels
"""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax(series: pd.Series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    print(res.summary())
    return res

def forecast(res, steps=12):
    fc = res.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()
