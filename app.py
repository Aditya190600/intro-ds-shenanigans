import pandas as pd
import numpy as np

import streamlit as st
import datetime

# Visualization Libraries
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
alt.data_transformers.disable_max_rows()

def comb(Chart2, Chart1):
    combine = (Chart2 | Chart1).configure_axis(
    grid=False).configure_view(
    strokeWidth=0).configure_title( 
    fontSize=15,
    anchor='middle',)
    return(st.altair_chart(combine) )

@st.cache
def datasetloader():
    data0 = pd.read_csv(source0)
    return data0

@st.cache
def dataset2():
    data2 = pd.read_csv(source2)
    return data2


st.markdown("# East Lansing Weather Analysis over 1979 to 2022")

if st.button("Project Analysis"):
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


The webapp will try to visualize the change in temprature and weather data and try to convey how this change in climate affects their lives and attempt in educating why various health and geological issues occur in East Lansing.  

For Visualization Tools, I primarily plan on using **Seaborn, Matplotlib**, **Altair**. In the beginning, Altair and HiPlot used to end up crashing my Code Editor (VS Code) and Jupyter Kernel due to Memory issue caused by the massive dataset and creates graphs of sizes in MB due to quantity of data.`

I feel this project is worthy of completion as 
1. It helps me learn and develop the **Art of Storytelling**. Data Science is all about going through the data and trying to weave a cohesive story out of what you learned from the data and explaining it to *either* **the people above** or **the masses**.    
1. It helps me in working on my Exploratory Data Analysis skills and helps me work on a wide array of tools and frameworks build by amazing personalities. eg. Pandas and Matplotlib.  
1. It helps me learn more on **Data Cleaning** and try to incorporate interesting concepts like **Data Imputing and Visualization** to give the Statistical Data another perspective in the *Digital Medium*  
    """)


source0 = "https://aimlessfox.s3.amazonaws.com/weather_EL.csv"
source2 = "https://aimlessfox.s3.amazonaws.com/Dropped_Weather_EL.csv"


if st.button("Dataset"):
    st.write("[Dataset Link](https://aimlessfox.s3.amazonaws.com/weather_EL.csv)")

data0 = datasetloader()

st.write("### View a part of the data")
st.write(data0.head(5))

st.markdown('# Data Preprocessing')
st.text('Visualizing NULL Values and removing data')

fig0 = plt.figure(figsize=(15, 10))
sns.heatmap(data0.isna().transpose(), cmap="rocket")
st.pyplot(fig0)

st.markdown(""" 
  As we look into the data, we realize there are **NaN Values** present in  **7** `out of the 26 columns`, namely
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

df_new = data0.drop(labels=["sea_level", "grnd_level", "wind_gust", "rain_1h", "rain_3h", "snow_1h", "snow_3h"], axis = 1)

data2 = dataset2()
data2.head()


st.markdown("# Correlation Matrix")
df_nll = df_nll = df_new.drop(labels =['lat','lon','city_name', 'timezone', 'weather_icon', 'dt'], axis = 1)

figc = plt.figure(figsize=(10,10))
sns.heatmap(df_nll.corr(), cmap = "rocket", annot= True)
st.pyplot(figc)

x_date = datetime.date(1979, 1, 1)
start_date = datetime.date(2021, 1, 1)
end_date = datetime.date(2022, 1, 1)

scale = alt.Scale(domain=['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm'],
                    range=['#e7ba52', '#ADD8E6', '#aec7e8', '#c7c7c7', '#E0DED7', '#CDD8D9', '#1f77b4', '#75766C', '#9467bd', '#004F63'])


start_gen, end_gen = st.slider(
    "What range do you want to visualize for boxplots of Weather in East Lansing??",
    value = (x_date, end_date)
)

st.write("You want to check for ", start_gen," to ", end_gen)

if st.button("GENERATE"):
    maskbar = (data2['dt_iso'] > str(start_gen)) & (data2['dt_iso'] <= str(end_gen))
    r = data2.loc[maskbar]
    maskchart = alt.Chart(r).mark_boxplot().encode(
        x=alt.X('temp:Q', scale=alt.Scale(domain=[240, 320])), 
        y="weather_main:N",
        color=alt.Color('weather_main:N' , legend=None, scale=scale),
    ).interactive(
    ).properties(
        title = 'Weather Watch over the Years',
        width=500, 
        height=400
    ).configure_axis(
    grid=False).configure_view(
    strokeWidth=0).configure_title( 
    fontSize=15,
    anchor='middle',)
    st.altair_chart(maskchart)


else:
    st.write("No Mask Generated. Graph will not be generated due to max data size being too huge")



if st.button("2021 vs 1979 analysis"):
    mask = (data2['dt_iso'] > str(start_date)) & (data2['dt_iso'] <= str(end_date))
    range_data = data2.loc[mask]

    
    Chart1 = alt.Chart(range_data).mark_point().encode(
        x='month(dt_iso):T',
        y=alt.Y('temp_min:Q', scale=alt.Scale(domain=[240, 320])),
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
        x='month(dt_iso):T',
        y=alt.Y('temp_min:Q', scale=alt.Scale(domain=[240, 320])),
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'temp_min:Q' ],
    ).interactive(
    ).properties(
        title = 'Weather Change over the Year 1979 for Minimum Temp',
        width=400, 
        height=400
    )

    comb(Chart2, Chart1)

    Histmin2021 = alt.Chart(range_data).mark_bar().encode(
        x='month(dt_iso):N',
        y='count()',
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'count()' ],
    ).properties(
        title = 'Weather Instances Count over the Year 2021',
        width=400, 
        height=400
    )

    Histmin1979 = alt.Chart(range_data_old).mark_bar().encode(
        x='month(dt_iso):N',
        y='count()',
        color=alt.Color('weather_main:N', legend=alt.Legend(title='Weather Type'), scale=scale),
        tooltip = ['weather_main:N', 'count()' ],
    ).properties(
        title = 'Weather Instances Count over the Year 1979',
        width=400, 
        height=400
    )

    comb(Histmin1979, Histmin2021)


    c1 = alt.Chart(data2).mark_line().encode(
        x='year(dt_iso):T',
        y=alt.Y('min(temp_min)', scale=alt.Scale(domain=[240, 320])),
        
    ).properties(
        title = 'Averge Minimum Temperatures over the Years',
        width=200, 
        height=200
    )

    c2 = alt.Chart(data2).mark_line().encode(
        x='year(dt_iso):T',
        y=alt.Y('mean(temp)', scale=alt.Scale(domain=[240, 320])),
    ).properties(
        title = 'Averge Temperatures over the Years',
        width=200, 
        height=200
    )

    c3 = alt.Chart(data2).mark_line().encode(
        x='year(dt_iso):T',
        y=alt.Y('max(temp_max)', scale=alt.Scale(domain=[240, 320])),
    ).properties(
        title = 'Averge Maximum Temperatures over the Years',
        width=200, 
        height=200
    )

    temp_ranges_over = (c1 | c2 | c3).configure_axis(
        grid=False).configure_view(
        strokeWidth=0).configure_title( 
        fontSize=15,
        anchor='middle',)

    st.altair_chart(temp_ranges_over)

