import pandas as pd
import numpy as np

import streamlit as st
import datetime

# Visualization Libraries
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


st.set_page_config(page_title="Regression Analysis", 
                   page_icon="icons/regression.png",
                   layout="wide",
                   initial_sidebar_state="auto")
st.write("""# East Lansing Weather Analysis over 1979 to 2022""")

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

## Library Imports
