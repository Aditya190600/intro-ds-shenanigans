import pandas as pd
import numpy as np

import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
import datetime

# Visualization Libraries
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

#Scaling libraries
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder


## Classification Algorithms Libraries
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, confusion_matrix, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

## Regression Algorithms Libraries
from sklearn.linear_model import LogisticRegression,LinearRegression 
from sklearn.model_selection import train_test_split #Splitting of Dataset


pd.set_option('display.max_columns', None)
alt.data_transformers.disable_max_rows()



show_pages(
        [

            Page(path="pages/1_📖_Problem_Statement.py",
                 name="Problem Statement",
                 icon="📖", ),

            Page(path="pages/2_📈_Data_Analysis_and_Visualization.py",
                 name="Data Analysis and Visualizations",
                 icon="📈", ),


            Section(name="Machine Learning and Stuff",
                    icon="🤖"
                    ),
            Page(path="pages/4_🧐_Regression_Analysis.py",
                 name="Regression Analysis",
                 icon="🧐", 
                 in_section=True),

            Page(path="pages/5_🧐_Classification_Analysis.py",
                 name="Classification Analysis",
                 icon="🧐", 
                 in_section=True),

            Page(path="pages/6_🧐_Time_Series_Analysis.py",
                 name="Time-Series Analysis",
                 icon="🧐", 
                 in_section=True),

            Page(path="pages/7_Predictions.py",
                name="Predictions",
                icon="📊", 
                in_section=True),

            Page(path="pages/8_Final_Results.py",
                name="Final Results and Conclusion",
                icon="🏁", 
                in_section=False),


        ]
    )

add_page_title(
                page_title="East Lansing Weather Analysis",
                page_icon="🌦️",
                layout="wide",
                initial_sidebar_state="auto")

