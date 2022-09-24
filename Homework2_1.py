
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly_express as px


st.write(""" 
# Iris Flower Dataset
How are Sepal Length, Sepal Width and Petal Length related to species size.
""")

df = sns.load_dataset("iris")

st.header("Data Stats")
st.write(df.describe())


fig = px.scatter_3d(df, x ='sepal_length', y="sepal_width", z='petal_length', color='species')

st.plotly_chart(fig)