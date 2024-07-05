import pandas as pd
import numpy as np

import streamlit as st

from st_pages import add_page_title

add_page_title(page_title="Regression Analysis", 
                   page_icon="icons/regression.png",
                   layout="wide",
                   initial_sidebar_state="auto")

import datetime

# Visualization Libraries
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Regression Libraries
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
# Load and Cache Datasets

@st.cache_data
def datasetloader():
    data0 = pd.read_csv(source0, index_col=[0])
    return data0

@st.cache_data
def dataset1():
    data1 = pd.read_csv(source1, index_col=[0])
    return data1

@st.cache_data
def dataset2():
    data2 = pd.read_csv(source2, index_col=[0])
    return data2


source0 = "data/weather_EL.csv"
source1 = "data/Cleaned_Weather_EL.csv"
source2 = "data/Dropped_Weather_EL.csv"


data1 = dataset1()

category_encode = {"weather_main": {'Fog': 0, 'Clouds': 1, 'Snow': 2, 'Clear': 3, 'Haze': 4, 'Drizzle': 5, 'Rain': 6,
                                'Thunderstorm': 7, 'Smoke': 8, 'Dust': 9, 'Mist': 10, 'Squall': 11} 
                }

category_recode = {"weather_main": {0: 'Fog',1: 'Clouds',2: 'Snow',3: 'Clear',4: 'Haze',5: 'Drizzle',6: 'Rain',
                                7: 'Thunderstorm',8: 'Smoke',9: 'Dust',10: 'Mist',11: 'Squall'} 
                }

st.write(" Here, we are performing som regression analysis for 4 Independent Variables(X) vs. temperature.")
st.write(" The 4 variables are **Pressure, Humidity, Wind Speed and Wind Degree**")
st.write("We should also note that, other than relative temperature fields and visibility, Temp shows negative correlation with most other fields.")
st.write("#### Linear Regression")
x = data1.iloc[:, [7, 8, 9, 10]]
y = data1.iloc[:, 1]
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=.3, random_state=15)

if(st.button( "Predict Linear Regression")):
    reg=LinearRegression().fit(x_train,y_train)
    y_pred=reg.predict(x_test)

    st.write("Using 3 of the columns for x and y as temp, we get")
    st.write(r2_score(y_test,y_pred) )

st.write("#### Decision Tree Regression")

max_depth_DT = st.slider('Max Depth', 2, 20, 10)
if(st.button( "Predict Decision Tree Regression")):
    DTree=DecisionTreeRegressor( random_state=15,max_depth=max_depth_DT)
    DTree.fit(x_train,y_train)
    y_predict=DTree.predict(x_test)

    st.write(r2_score(y_test,y_predict))

st.write("#### KNN Regression")
neighbor_count = st.slider('Number of Neighbors', 5, 30, 25)
if(st.button( "Predict KNN Regression")):
    KnnRegressor=KNeighborsRegressor(n_neighbors=neighbor_count)
    KnnRegressor.fit(x_train,y_train)
    y_predict=KnnRegressor.predict(x_test)
    st.write(r2_score(y_test,y_predict))