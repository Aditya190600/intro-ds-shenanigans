import pandas as pd
import numpy as np

import streamlit as st
import datetime
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error

from st_pages import add_page_title

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


add_page_title(page_title="Time-Series Analysis", 
                   page_icon="icons/timeseries.png",
                   layout="wide",
                   initial_sidebar_state="auto")

# Importing Libraries

# Time Series Libraries

from prophet.serialize import model_from_json

with open('models/Time-Series/Prophet_model.json', 'r') as fin:
    model = model_from_json(fin.read())  # Load model

# Load Imputed Data
df_imputed = pd.read_csv('data/Imputed_Weather_EL.csv')
df_imputed['dt_iso'] = pd.to_datetime(df_imputed['dt_iso'], format='%Y-%m-%d %H:%M:%S+00:00')
df_imputed['dt_iso'] = df_imputed['dt_iso'].dt.tz_localize(None)
df_imputed.set_index('dt_iso', inplace=True)

# st.write(df_imputed.head())

X = df_imputed.drop(columns=['temp', 'temp_min', 'temp_max', 'feels_like'])
y = df_imputed['temp']


# Print the Train y 
# Streamlit plotly chart


## Edit done to Streamlit Markdown to center images when kept in full screen
st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

col1, col2, col3 = st.columns([.5, .75, .5])
with col1:
    st.write(' ')
with col2:
    st.plotly_chart(px.line(y, title='Temperature Time Series'))
with col3:
    st.write(' ')


X_test = pd.read_csv('data/timeseries/test_data.csv')
X_test.set_index('dt_iso', inplace=True)

y_test = pd.read_csv('data/timeseries/y_test_data.csv')
y_test.set_index('dt_iso', inplace=True)


# Create a DataFrame compatible with Prophet for the test set
future = pd.DataFrame({'ds': X_test.index})
for col in X_test.columns:
    future[col] = X_test[col].values
# Make predictions
forecast = model.predict(future)

# Convert 'ds' column to Timestamp type
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Ensure the 'ds' column is set as the index
forecast = forecast.set_index('ds')

forecast.head()

# Calculate RMSE
rmse = mean_squared_error(y_test, forecast['yhat'])
mape = mean_absolute_percentage_error(y_test, forecast['yhat'])
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Abosulte Percentage Error (MAPE): {mape}')

# Plot actual vs. predicted using Plotly
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))
fig1.add_trace(go.Scatter(x=y_test.index, y=forecast['yhat'], mode='lines', name='Predicted', line=dict(dash='dash', color='orange')))

fig1.update_layout(title='Demand Forecasting with Prophet',
                  xaxis_title='Date',
                  yaxis_title='Demand',
                  legend=dict(x=0, y=1, traceorder='normal'))

st.plotly_chart(fig1)

# Plot actual vs. predicted using Plotly with confidence interval
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))

# Predicted data with confidence interval
fig.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat'], mode='lines', name='Predicted',
                         line=dict(dash='dash', color='orange')))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_upper'], fill='tonexty', mode='lines', name='Upper CI',
                         line=dict(dash='dash', color='rgba(255, 165, 0, 0.2)')))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast['yhat_lower'], fill='tonexty', mode='lines', name='Lower CI',
                         line=dict(dash='dash', color='rgba(255, 165, 0, 0.2)')))

fig.update_layout(title='Demand Forecasting with Prophet and Confidence Interval',
                  xaxis_title='Date',
                  yaxis_title='Demand',
                  legend=dict(x=0, y=1, traceorder='normal'))

st.plotly_chart(fig)