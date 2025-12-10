import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
# import warnings

# BO function 
def optimize_sarimax_params(X_train, y_train, m = 7):

    # Objective Function
    # warnings.filterwarnings("ignore")
    model = pm.auto_arima(
        y=y_train,
        X=X_train,
        start_p=0, max_p=3, # avoid overfitting
        start_q=0, max_q=3,
        m=m,                # Seasonal Period
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        seasonal=True,      # finding sp
        d=None,              
        D=None,             
        trace=False,        
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True,      # True = Smart search, False = Grid search
        information_criterion='aic', # penalty
        n_jobs=-1           # maximize cpu thread 
    )
    #For debug
    #print(model.summary())
    return model.order, model.seasonal_order
