# East Lansing Weather Analysis Project

## 🎯 Project Title
**East Lansing Weather Pattern Analysis (1979-2022): A Comprehensive Data Science Approach**

## 📖 Overview
A comprehensive data science project analyzing 40+ years of weather patterns in East Lansing, Michigan using advanced machine learning techniques. This interactive web application visualizes climate change trends, predicts weather patterns, and educates users about the environmental impact of human population growth on local weather systems.

## 🔍 Problem Statement
- **Challenge**: Understanding how weather patterns have changed in East Lansing over the past 43 years and identifying potential correlations with human population growth
- **Importance**: Climate change analysis is crucial for understanding environmental impacts and planning for future weather-related challenges
- **Beneficiaries**: Meteorologists, environmental researchers, policy makers, and the general public interested in local climate trends

## 🎨 Features & Functionality
- **Interactive Data Visualization**: Dynamic charts and graphs showing weather trends over time
- **Machine Learning Predictions**: Weather forecasting using regression, classification, and time-series models
- **Comprehensive EDA**: Detailed exploratory data analysis with correlation matrices and statistical insights
- **Multi-Model Comparison**: Performance comparison across different ML algorithms
- **Web Interface**: User-friendly Streamlit application with multiple analysis sections
- **Data Imputation**: Advanced techniques for handling missing weather data

## 🛠️ Technologies Used
### Programming Languages
- Python 3.11

### Libraries & Frameworks
- **Data Analysis**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, seaborn, plotly, altair
- **Machine Learning**: scikit-learn, tensorflow/keras, xgboost, lightgbm, catboost
- **Time Series**: Prophet, ARIMA, pmdarima
- **Web Framework**: streamlit, st_pages
- **Deep Learning**: TensorFlow/Keras for neural networks

### Tools & Platforms
- Jupyter Notebooks for analysis
- Git for version control
- OpenWeatherMap API (data source)

## 📊 Data & Methodology
- **Data Source**: OpenWeatherMap API historical weather data for East Lansing, Michigan
- **Data Size**: 414,847 hourly weather observations from 1979-2022 (43 years)
- **Variables**: 26 weather parameters including temperature, pressure, humidity, wind, precipitation, and cloud cover
- **Analysis Methods**: 
  - Exploratory Data Analysis with correlation analysis
  - Data cleaning and imputation using linear regression
  - Statistical modeling and hypothesis testing
  - Machine learning for regression and classification
  - Time-series forecasting

## 🎯 Key Insights & Results
### Model Performance
- **Best Regression Model**: Gradient Boosting (R² = 0.441)
- **Best Classification Model**: Random Forest (74% accuracy)
- **Time Series**: Prophet model with RMSE = 0.527, MAPE = 0.19%
- **Data Quality**: Successfully imputed 54,063 missing visibility values

### Weather Patterns Discovered
- **Dominant Weather Types**: Clouds (45.5%), Clear (19.1%), Rain (10.4%)
- **Temperature Trends**: Analyzed 43-year temperature patterns and seasonal variations
- **Correlation Insights**: Strong correlations between temperature variables and weather conditions

## 🚀 Live Demo
- **Web App**: Streamlit multi-page application with interactive visualizations
- **GitHub Repository**: Available with complete source code and documentation
- **Dataset**: Publicly available cleaned weather data

## 📈 Impact & Applications
- **Meteorologists**: Pattern recognition for long-term weather forecasting
- **Urban Planners**: Understanding climate impact for city development
- **Researchers**: Climate change analysis and environmental studies
- **Public Education**: Accessible climate data visualization for community awareness
- **Future Extensions**: Integration with population data, expansion to other cities

## 🔧 Technical Implementation
- **Architecture**: Modular Streamlit application with separate analysis pages
- **Data Pipeline**: CSV processing → Cleaning → Imputation → Model Training → Prediction
- **Model Management**: Serialized models saved as .pkl and .json files
- **Key Algorithms**: 
  - Linear regression for data imputation
  - Ensemble methods (Random Forest, Gradient Boosting)
  - Neural networks using Keras
  - Prophet for time-series forecasting
- **Challenges Solved**: Missing data handling, large dataset processing, model comparison framework

## 📱 Screenshots/Visuals
- **Correlation Heatmap**: Comprehensive correlation matrix of all weather variables
- **Temperature Distribution**: Box plots showing temperature patterns by weather type
- **Time Series Plots**: Interactive plotly visualizations of weather trends over time
- **Model Performance**: Comparison charts of different ML algorithm results

## 🏆 Achievements
- **Dataset Processing**: Successfully analyzed 400K+ weather observations
- **Model Diversity**: Implemented 15+ different machine learning algorithms
- **Visualization Excellence**: Created comprehensive interactive dashboard
- **Technical Depth**: Combined statistical analysis, ML, and time-series forecasting

## 🤝 Collaboration & Credits
- **Solo Project**: Complete end-to-end data science project
- **Data Source**: OpenWeatherMap API
- **Inspiration**: Climate change awareness and data storytelling
- **Special Thanks**: Michigan State University location (East Lansing) for local relevance

---
*This project demonstrates expertise in **data science, machine learning, time-series analysis, and web application development**, showcasing the ability to **transform raw weather data into actionable insights through comprehensive statistical analysis and predictive modeling**.*