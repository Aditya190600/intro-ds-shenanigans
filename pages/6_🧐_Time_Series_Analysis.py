import pandas as pd
import numpy as np

import streamlit as st
import datetime

from st_pages import add_page_title

add_page_title(page_title="Time-Series Analysis", 
                   page_icon="icons/timeseries.png",
                   layout="wide",
                   initial_sidebar_state="auto")