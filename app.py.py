import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
#from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
from autots import AutoTS

html_temp = """
<div style="background-color:fuchsia;padding:10px">
<h2 style="color:white;text-align:center;">Forecasting The Cement Sales </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)
st.text("")
uploaded_file = st.file_uploader(" ", type=['xlsx'])

if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['Month'] = cement['Month'].apply(lambda x: x.strftime('%B-%Y'))

    st.write("Plese wait for the forecasting result... Model is working on it")
    
    mod = AutoTS(forecast_length=12, frequency='M', prediction_interval = 0.90,
             ensemble= None, model_list = 'univariate', max_generations = 3, num_validations= 2,
             no_negatives = True,n_jobs = 'auto')

    mod = mod.fit(cement, date_col='Month', value_col='Sales')
    prediction = mod.predict()
    
    forecast = prediction.forecast
    
    st.subheader("From AutoTS model")
   
    st.write("Forecast of Sales for next 12 months: ", forecast)
