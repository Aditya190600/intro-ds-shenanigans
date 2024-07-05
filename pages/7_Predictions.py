import pandas as pd
import numpy as np

import streamlit as st
import datetime

from st_pages import add_page_title

add_page_title(page_title="Prediction", 
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="auto")


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

## Classification Algorithms Libraries
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

## Regression Algorithms Libraries
from sklearn.linear_model import LogisticRegression #Logistic Regression is a Machine Learning classification algorithm
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.model_selection import train_test_split #Splitting of Dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import plotly.express as px


@st.cache_data
def dataset1():
    data1 = pd.read_csv(source1, index_col=[0])
    return data1

source1 = "data/Cleaned_Weather_EL.csv"



st.write("Let's try to Predict some stuff, Shall we?")
data1 = dataset1()

category_encode = {"weather_main": {'Fog': 0, 'Clouds': 1, 'Snow': 2, 'Clear': 3, 'Haze': 4, 'Drizzle': 5, 'Rain': 6,
                                'Thunderstorm': 7, 'Smoke': 8, 'Dust': 9, 'Mist': 10, 'Squall': 11} 
                }

category_recode = {"weather_main": {0: 'Fog',1: 'Clouds',2: 'Snow',3: 'Clear',4: 'Haze',5: 'Drizzle',6: 'Rain',
                                7: 'Thunderstorm',8: 'Smoke',9: 'Dust',10: 'Mist',11: 'Squall'} 
                }

data1 = data1.replace(category_encode)
new_df = data1[data1['weather_main'].isin([0,1, 2, 3, 4, 5, 6, 7, 10])]

x = new_df.iloc[:, [1, 7, 8, 9]]
y = new_df.iloc[:, 13]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=15)

mod = st.selectbox('Select the model for prediction:.', ['KNN', 'Decision Tree', 'Random Forest'])

temp = st.slider(" Temperature (in K)", 260, 350, 275 )
pres = st.slider(" Pressure", 910, 1050, 950 )
humidity = st.slider(" Humidity", 910, 1050, 950 )
winspd = st.slider("Wind Speed", 0, 27, 15 )

pred = st.button("Predict")
if pred:
    if mod == 'KNN':
        model = KNeighborsRegressor(n_neighbors = 5,p=2)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    
    elif mod == 'Decision Tree':
        model = DecisionTreeRegressor(max_depth=10, random_state = 0)
        model.fit(x_train, y_train)

    elif mod == 'Random Forest':
        model = RandomForestRegressor(n_estimators = 15, random_state = 0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    value_x = (model.predict([[temp, pres,humidity,winspd]])[0])
    st.write("The Weather:  ", category_recode['weather_main'][int(value_x)])
