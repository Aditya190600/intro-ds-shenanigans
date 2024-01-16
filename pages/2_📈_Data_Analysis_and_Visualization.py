import pandas as pd
import numpy as np
import os
import streamlit as st
import datetime

# Visualization Libraries
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


st.set_page_config(page_title="Data Analysis and Visualizations", 
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="auto")

st.write("""
# East Lansing Weather Analysis over 1979 to 2022
""")

@st.cache_data(persist= True)
def comb(ChartB, ChartA):
    combine = (ChartB | ChartA).configure_axis(
    grid=False).configure_view(
    strokeWidth=0).configure_title( 
    fontSize=15,
    anchor='middle')
    st.altair_chart(combine)
    return()

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

st.markdown("### Correlation Matrix for the new dataframe")
st.image("images/dfcorr.png")
st.markdown(""" With the correlation matrix given above, we can see that 
**`Temp`**, **`Temp_min`** and **`Temp_max`** have the greatest correlation value of **1**.  
**`Feels_like`** and **`Temp`** have the seecond strongest positive correlation of **0.99**   
The greatest negative correlation lies with **`Humidity`** and **`Visibility`** which is **-0.48**  
The second lowest negative correlation is between **`Clouds_all`** and **`Weather_id`** or  
**`Clouds_all`** and **`Visibility`** which is **-0.36**.   
A possible reason for the strong negative correlation between Humidity and Visibility might be due 
to the presence of Null Values in Visibility column.
""")

x_date = datetime.date(1979, 1, 1)
start_date = datetime.date(2021, 1, 1)
end_date = datetime.date(2022, 1, 1)

scale = alt.Scale(domain=['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm'],
                    range=['#e7ba52', '#ADD8E6', '#aec7e8', '#c7c7c7', '#E0DED7', '#CDD8D9', '#1f77b4', '#75766C', '#9467bd', '#004F63'])


if st.checkbox("Box plots for visualiztions over the Years"):
    start_gen, end_gen = st.slider(
        "What range do you want to visualize for boxplots of Weather in East Lansing??",
        value = (x_date, end_date)
    )

    data2 = dataset2()

    st.write("You want to check for ", start_gen," to ", end_gen)
    options = st.multiselect("Choose your weather", data2["weather_main"].unique(), [])

    if st.button("GENERATE"):
        maskbar = (data2['dt_iso'] > str(start_gen)) & (data2['dt_iso'] <= str(end_gen)) & (data2['weather_main'].isin(options))
        r = data2.loc[maskbar]


        maskchart = alt.Chart(r).mark_boxplot().encode(
            x=alt.X('temp:Q', scale=alt.Scale(domain=[240, 320]), title='Temperature(in K)'), 
            y=alt.Y("weather_main:N", title="Weather Type"),
            color=alt.Color('weather_main:N' , legend=None, scale=scale),
        ).interactive(
        ).properties(
            title = 'Weather Watch over the Years',
            height = 500,
            width=600, 
        ).configure_axis(
        grid=False).configure_view(
        strokeWidth=0).configure_title( 
        fontSize=15,
        anchor='middle',)
        st.altair_chart(maskchart)



data2 = dataset2()

if st.checkbox("1979 vs 2021 analysis"):
    mask = (data2['dt_iso'] > str(start_date)) & (data2['dt_iso'] <= str(end_date))
    range_data = data2.loc[mask]

    
    Chart1 = alt.Chart(range_data).mark_point().encode(
        x=alt.X('month(dt_iso):T', title='Months'),
        y=alt.Y('temp_min:Q', scale=alt.Scale(domain=[240, 320]), title='Minimum Temperature'),
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'temp_min:Q' ],
    ).interactive(

    ).properties(
        title = 'Weather Change over the Year 2021 for Minimum Temp',
        width=400, 
        height=400
    )

    start_date_old = datetime.date(1979, 1, 1)
    end_date_old = datetime.date(1980, 1, 1)
    mask_old = (data2['dt_iso'] > str(start_date_old)) & (data2['dt_iso'] <= str(end_date_old))

    range_data_old = data2.loc[mask_old]

    Chart2 = alt.Chart(range_data_old).mark_point().encode(
        x=alt.X('month(dt_iso):T', title='Months'),
        y=alt.Y('temp_min:Q', scale=alt.Scale(domain=[240, 320]), title='Minimum Temperature'),
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'temp_min:Q' ],
    ).interactive(
    ).properties(
        title = 'Weather Change over the Year 1979 for Minimum Temp',
        width=400, 
        height=400
    )
    comb(Chart2, Chart1)
    st.markdown("""With the Scatterplots for 1979 and 2021, we notice that: 
1. The Coldest days of the Years were **January and February** for 1979 but had changed to only **February** in 2021.  
2.  The lowest recorded Temperatures were *245.98 K* for 1979 and *245.02 K* for 2021.    
3. The Hottest Days of the Years were consistent with **August** in case both the years, where the Temperatures were around *305 K*.   
    """)

if st.checkbox("HistPlot 1979 vs 2021 analysis"):
    mask = (data2['dt_iso'] > str(start_date)) & (data2['dt_iso'] <= str(end_date))
    range_data = data2.loc[mask]

    
    ChartH2021 = alt.Chart(range_data).mark_bar().encode(
        x=alt.X('month(dt_iso):T', title='Months'),
        y='count()',
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'count()', 'mean(temp)' ],
    ).interactive(

    ).properties(
        title = 'Weather Instances Count over the Year 2021',
        width=400, 
        height=400
    )

    start_date_old = datetime.date(1979, 1, 1)
    end_date_old = datetime.date(1980, 1, 1)
    mask_old = (data2['dt_iso'] > str(start_date_old)) & (data2['dt_iso'] <= str(end_date_old))

    range_data_old = data2.loc[mask_old]

    ChartH1979 = alt.Chart(range_data_old).mark_bar().encode(
        x=alt.X('month(dt_iso):T', title='Months'),
        y='count()',
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'count()', 'mean(temp)' ],
    ).interactive(
    ).properties(
        title = 'Weather Instances Count over the Year 1979',
        width=400, 
        height=400
    )
    comb(ChartH1979, ChartH2021)
    st.markdown("""With the Histplots for 1979 and 2021, we notice that: 
1. There is a drasctic drop in the presence of Clouds over the years. From around 350 records (350/24 = 14 days )average, it had dropped to a 200 record average.  
2.  We also notice that the cases for Clear Skies have increased drastically.    
3. Foggy and Haze Weather climate had become Mist.   
4. Snowy weather had become Rainy climate.
    """)