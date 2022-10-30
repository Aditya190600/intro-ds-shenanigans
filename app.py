from turtle import title
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

@st.cache(suppress_st_warning=True)
def comb(ChartB, ChartA):
    combine = (ChartB | ChartA).configure_axis(
    grid=False).configure_view(
    strokeWidth=0).configure_title( 
    fontSize=15,
    anchor='middle')
    st.altair_chart(combine)
    return()


@st.cache
def datasetloader():
    data0 = pd.read_csv(source0)
    return data0

@st.cache
def dataset1():
    data1 = pd.read_csv(source1)
    return data1

@st.cache
def dataset2():
    data2 = pd.read_csv(source2)
    return data2


st.markdown("# East Lansing Weather Analysis over 1979 to 2022")

st.write("⚠️ **STATUATORY WARNING** Some graphs might not load in Firefox Browser. You are kindly requested to view using an alternative browser.")
if st.checkbox("Project Analysis"):
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

\\

The webapp will try to visualize the change in temprature and weather data and try to convey how this change in climate affects their lives and attempt in educating why various health and geological issues occur in East Lansing.  

For Visualization Tools, I primarily plan on using **Seaborn, Matplotlib**, **Altair**. In the beginning, Altair and HiPlot used to end up crashing my Code Editor (VS Code) and Jupyter Kernel due to Memory issue caused by the massive
dataset and creates graphs of sizes in MB due to quantity of data.

I feel this project is worthy of completion as 
1. It helps me learn and develop the **Art of Storytelling**. Data Science is all about going through the data and trying to weave a cohesive story out of what you learned from the data and explaining it to *either* **the people above** or **the masses**.    
1. It helps me in working on my Exploratory Data Analysis skills and helps me work on a wide array of tools and frameworks build by amazing personalities. eg. Pandas and Matplotlib.  
1. It helps me learn more on **Data Cleaning** and try to incorporate interesting concepts like **Data Imputing and Visualization** to give the Statistical Data another perspective in the *Digital Medium*  
""")


source0 = "https://aimlessfox.s3.amazonaws.com/weather_EL.csv"
source1 = "https://aimlessfox.s3.amazonaws.com/Cleaned_Weather_EL.csv"
source2 = "https://aimlessfox.s3.amazonaws.com/Dropped_Weather_EL.csv"

if st.button("Dataset"):
    st.write("[Dataset Link](https://aimlessfox.s3.amazonaws.com/weather_EL.csv)")

data0 = datasetloader()

if st.checkbox("View a part of the data"):
    st.write(data0.head(5))

if st.checkbox("Describe the data"):
    st.write(data0.describe() )

if st.checkbox("View Null Values"):
    st.image("images/nullval.png")
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

if st.checkbox("View Correlation Matrix for Transformed dataset"):
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


if st.checkbox("Conclusion"):
    c1 = alt.Chart(data2).mark_line().encode(
        x=alt.X('year(dt_iso):T', title='Years'),
        y=alt.Y('min(temp_min)', scale=alt.Scale(domain=[240, 320]), title='Minimum Temperature'),
        
    ).properties(
        title = 'Minimum Temperatures over the Years',
        width=200, 
        height=200
    )

    c2 = alt.Chart(data2).mark_line().encode(
        x=alt.X('year(dt_iso):T', title='Years'),
        y=alt.Y('mean(temp)', scale=alt.Scale(domain=[240, 320]), title='Mean Temperature'),
    ).properties(
        title = 'Temperatures over the Years',
        width=200, 
        height=200
    )

    c3 = alt.Chart(data2).mark_line().encode(
        x=alt.X('year(dt_iso):T', title='Years'),
        y=alt.Y('max(temp_max)', scale=alt.Scale(domain=[240, 320]), title='Maximum Temperature'),
    ).properties(
        title = 'Maximum Temperatures over the Years',
        width=200, 
        height=200
    )

    temp_ranges_over = (c1 | c2 | c3).configure_axis(
        grid=False).configure_view(
        strokeWidth=0).configure_title( 
        fontSize=15,
        anchor='middle',)

    st.altair_chart(temp_ranges_over)
    st.markdown(""" ### Conclusion  
We notice with the above graphs that there have been many changes in weather patterns over the past years.  
These patterns could be related to various underlying factors. The primary cause could be attributed to **Climate Change** and **Global Warming**.  
But also there is a increase in the Minimum Temperatures and a drop in Maximum Temperatures post 2020. This could be attributed to the COVID Pandemic Lockdowns.

""")
