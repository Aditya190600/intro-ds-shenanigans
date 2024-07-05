import pandas as pd
import numpy as np

import streamlit as st
import datetime

from st_pages import add_page_title

add_page_title(page_title="Classification Analysis", 
                   page_icon="icons/classification.png",
                   layout="wide",
                   initial_sidebar_state="auto")

# Importing Libraries

#Scaling libraries
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

data1 = dataset1()

source0 = "data/weather_EL.csv"
source1 = "data/Cleaned_Weather_EL.csv"
source2 = "data/Dropped_Weather_EL.csv"


category_encode = {"weather_main": {'Fog': 0, 'Clouds': 1, 'Snow': 2, 'Clear': 3, 'Haze': 4, 'Drizzle': 5, 'Rain': 6,
                                'Thunderstorm': 7, 'Smoke': 8, 'Dust': 9, 'Mist': 10, 'Squall': 11} 
                }

category_recode = {"weather_main": {0: 'Fog',1: 'Clouds',2: 'Snow',3: 'Clear',4: 'Haze',5: 'Drizzle',6: 'Rain',
                                7: 'Thunderstorm',8: 'Smoke',9: 'Dust',10: 'Mist',11: 'Squall'} 
                }


@st.cache_resource
def randForest_classifier(X_train, X_test, y_train, y_test):
    classifier_RF = RandomForestClassifier(criterion='entropy', random_state=0)
    classifier_RF.fit(X_train, y_train)
    y_pred = classifier_RF.predict(X_test)
    
    print (r2_score(y_test,y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier_RF.classes_)
    disp.plot()
    plt.show()  
    
    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)


@st.cache_resource
def dTree_classifier(X_train, X_test, y_train, y_test):
    #classifier_svm = SVC(kernel='rbf', random_state=0)
    classifier_dTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier_dTree.fit(X_train, y_train)
    y_pred = classifier_dTree.predict(X_test)
    
    print (accuracy_score(y_test,y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier_dTree.classes_)
    disp.plot()
    plt.show()  
    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)

counter = data1.groupby('weather_main').count()['dt_iso'].rename_axis("weather_main").reset_index(name="weather_count")

labels = counter.weather_main
values = counter.weather_count
st.write(counter)
st.write("Over here, we can see that **Dust, Smoke and Squall** have very low counts of occurance and hence are removed.")

data1 = data1.replace(category_encode)
new_df = data1[data1['weather_main'].isin([0,1, 2, 3, 4, 5, 6, 7, 10])]

x = new_df.iloc[:, [1, 2, 7, 8, 9]]
y = new_df.iloc[:, 13]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=15)

type = st.radio(
"Select type of algorithm",
("KNN", "Decision Tree","Random Forest")
)
if(type=="KNN"):
    st.write("K-nearest Neighbour")

    param1 = st.multiselect('Select features',['temp','humidity', 'pressure','wind_speed'],['temp'])

    neighbor_count = st.slider('Number of Neighbors', 2, 30, 5)

    knn = KNeighborsClassifier(metric='minkowski', n_neighbors = neighbor_count, p=2)
    knn.fit(x_train.loc[:,param1], y_train)
    y_pred = knn.predict(x_test.loc[:,param1])
    accuracy = accuracy_score(y_test,y_pred)

    st.write("Accuracy")
    st.write(accuracy)

    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)

    st.write("Confusion Matrix")    
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, origin='lower',text_auto=True))

if(type=="Decision Tree"):
    st.write("Decision Tree Classifier")
    
    param2 = st.multiselect('Select features',['temp', 'pressure','humidity','wind_speed'],['temp'])

    max_depth_DT = st.slider('Max Depth', 2, 15, 10)

    classifier_dTree = DecisionTreeClassifier(criterion='entropy', random_state=0,max_depth=max_depth_DT)
    classifier_dTree.fit(x_train.loc[:,param2], y_train)
    y_pred = classifier_dTree.predict(x_test.loc[:,param2])
    accuracy = accuracy_score(y_test,y_pred)
    st.write("Accuracy")
    st.write(accuracy)

    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)

    st.write("Confusion Matrix")    
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm,color_continuous_scale='Viridis', origin='lower',text_auto=True))    

if(type=="Random Forest"):
    st.write("Random Forest Classifier")

    param3 = st.multiselect('Select features',['temp', 'pressure','humidity','wind_speed'],['temp'])

    n_estimators_RF = st.slider('N Estimators', 2, 100, 10)

    classifier_RF = RandomForestClassifier(criterion='entropy', random_state=0,n_estimators=n_estimators_RF)
    classifier_RF.fit(x_train.loc[:,param3], y_train)
    y_pred = classifier_RF.predict(x_test.loc[:,param3])
    accuracy = accuracy_score(y_test,y_pred)
    st.write("Accuracy")
    st.write(accuracy)            #st.write(classification_report(y_train,knn.predict(y_train.loc[:,param])))
    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)
    st.write("Confusion Matrix")    
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm,color_continuous_scale='Viridis', origin='lower',text_auto=True))    