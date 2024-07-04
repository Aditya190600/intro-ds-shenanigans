import pandas as pd
import numpy as np

import streamlit as st

from PIL import Image

from st_pages import add_page_title

add_page_title(page_title="Problem Statement",
                page_icon="🌦️",
                layout="wide",
                initial_sidebar_state="auto")



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

st.write("""
# East Lansing Weather Analysis over 1979 to 2022
""")

source0 = "data/weather_EL.csv"
source1 = "data/Cleaned_Weather_EL.csv"
source2 = "data/Dropped_Weather_EL.csv"


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
 

image1 = Image.open("images/meme.png")

col1, col2, col3 = st.columns([1, .6, 1])
with col1:
    st.write(' ')
with col2:
    st.image(image1, caption="Let's start with some facts about Weather ...", width=150, use_column_width = True)
with col3:
    st.write(' ')

st.markdown("## Project Analysis")
st.markdown('Our primary goal in this project is to **`Analyze the change in weather patterns over the period from 1979 to 2022 and see if there are any changes in weather patterns over East Lansing.  '
        '`** The big picture is it can be co-related with the population change of the area to see the effects of the human population on nature. '
        'This project is useful for **`Meteorologists`** as it can see if any weather patterns have been formed over the last 40 years and if there are any cycles.'
        ' It can also be used to understand how human life has an impact on the surrounding climate and predict if the impact might have more adverse results in the future.  '  
        )
st.markdown(
    """This dataset contained a lot of variables and felt more explorable for a Project. This dataset contains around 26 Variables (as listed below).  

| No. | Name | Type |
|---|---|---|
| 1. | **`dt`** | Time of data calculation, unix, `UTC` |
| 2. | **`dt_iso`** | Date and time in UTC format |
| 3. | **`timezone`** | Shift in seconds from UTC |
| 4. | **`city_id`** | ZIP code |
| 5. | **`city_name`** | City name |
| 6. | **`lat`** | Geographical coordinates of the location `(latitude)` |
| 7. | **`lon`** | Geographical coordinates of the location `(longitude)` |
| 8. | **`temp`** | Temperature, `Kelvin` |
| 9. | **`feels_liken`** | This temperature parameter accounts for the human perception of weather, `Kelvin` |
| 10. | **`temp_min`** | Minimum temperature at the moment. This is deviation from temperature that is possible for large cities and megalopolises geographically expanded (use these parameter optionally), `Kelvin` |
| 11. | **`temp_max`** | Maximum temperature at the moment. This is deviation from temperature that is possible for large cities and megalopolises geographically expanded (use these parameter optionally), `Kelvin` |
| 12. | **`pressure`** | Atmospheric pressure (on the sea level), `hPa` |
| 13. | **`sea_level`** | Sea level pressure, `hPa` |
| 14. | **`grnd_level`** | Ground level pressure, `hPa` |
| 15. | **`humidity`** | Humidity, `%` |
| 16. | **`wind_speed`** | Wind speed, `meter/sec` |
| 17. | **`wind_deg`** | Wind direction, `degrees` (meteorological) |
| 18. | **`rain_1h`** | Rain volume for the last hour, `mm` |
| 19. | **`rain_3h`** | Rain volume for the last 3 hours, `mm` |
| 20. | **`snow_1h`** | Snow volume for the last hour, `mm (in liquid state)` |
| 21. | **`snow_3h`** | Snow volume for the last 3 hours, `mm (in liquid state)` |
| 22. | **`clouds_all`** | Cloudiness, `%` |
| 23. | **`weather_id`** | Weather condition id (more info Weather condition codes) |
| 24. | **`weather_main`** | Group of weather parameters (Rain, Snow, Extreme, etc.) |
| 25. | **`weather_description`** | Weather condition within the group |
| 26. | **`weather_icon`** | Weather icon id (more info Weather icon) |  
""" )
st.markdown("## Problem Statement")

st.markdown(
"""
The webapp will try to visualize the change in temprature and weather data and try to convey how this change in climate affects their lives and attempt in educating why various health and geological issues occur in East Lansing.  

For Visualization Tools, I primarily plan on using **Seaborn, Matplotlib**, **Altair**. In the beginning, Altair and HiPlot used to end up crashing my Code Editor (VS Code) and Jupyter Kernel due to Memory issue caused by the massive
dataset and creates graphs of sizes in MB due to quantity of data.

I feel this project is worthy of completion as 
1. It helps me learn and develop the **Art of Storytelling**. Data Science is all about going through the data and trying to weave a cohesive story out of what you learned from the data and explaining it to *either* **the people above** or **the masses**.    
1. It helps me in working on my Exploratory Data Analysis skills and helps me work on a wide array of tools and frameworks build by amazing personalities. eg. Pandas and Matplotlib.  
1. It helps me learn more on **Data Cleaning** and try to incorporate interesting concepts like **Data Imputing and Visualization** to give the Statistical Data another perspective in the *Digital Medium*  
    """)



st.link_button(label= "Dataset", 
                    url = "https://raw.githubusercontent.com/Aditya190600/intro-ds-shenanigans/main/data/weather_EL.csv", 
                    help = "The repository for the original dataset used in this project", 
                    type = "secondary")
data0 = datasetloader()

if st.checkbox("View a part of the data"):
    st.write(data0.head(5))

if st.checkbox("Describe the data"):
    st.write(data0.describe() )

if st.checkbox("View Null Values"):
    st.image("images/nullval.png")
    st.markdown(""" 
As we look into the data, we realize there are **`NaN Values`** present in  **7** `out of the 26 columns`, namely
1. **`"sea_level"`**
2.  **`"grnd_level"`**
3.  **`"wind_gust"`**
4.  **`"rain_1h"`**
5.  **`"rain_3h"`**
6.  **`"snow_1h"`**
7.  **`"snow_3h"`**,  

while 5 other variables, 

1.  **`'lat'`** = 42.736979	
2.   **`'lon'`** = 	-84.483865
3.   **`'city_name'`** = East Lansing
4.   **`'timezone'`** = -18000, -14400 (This is due to the change of timezones post *1931* due to Detroit)
5.   **`'weather_icon'`** = This is just information encoding to showcase in the OpenWeatherMap app  

had to be removed due to presence of a single unique value for the entire dataset.  

In terms of **Missing Data and Missingness**, as mentioned above, we have 7 columns, which are *either* `completly NULL`, *or*, `more than 95% of the data` is missing. So this primarily falls under **No data or MCAR**. 

""")
