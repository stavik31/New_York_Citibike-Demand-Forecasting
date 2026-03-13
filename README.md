# New York CitiBike Demand Forecasting

## Description

This project forecasts hourly bike rental demand for New York City's CitiBike system using five years of trip data (2020–2025) merged with historical weather data. The analysis covers the full data science pipeline: data collection, preprocessing, exploratory analysis (trend, seasonality, anomaly detection, weather correlation), and demand forecasting. A broad set of forecasting models is implemented and compared — from simple statistical baselines (Mean, Naive, Drift) to classical time series models (SES, Holt-Winters, ARIMA, SARIMA, ARIMAX, SARIMAX) and deep learning models (MLP, Multi-headed MLP, CNN, RNN, LSTM). The best performing model was a Multivariate MLP with an R² of ~84%, demonstrating that incorporating weather features (temperature, precipitation, humidity, wind speed, etc.) significantly improves demand forecasting accuracy.

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn torch requests
```

### 2. Collect Raw Data

**CitiBike trip data:**
- Download monthly CitiBike CSV files from [CitiBike System Data](https://citibikenyc.com/system-data) for January 2020 through April 2025
- Place each month's file(s) inside a folder named by month (e.g. `bike_data/202001-citibike-tripdata/`)

**Weather data:**
- Open and run `weather_data_fetch.ipynb`
- This fetches hourly weather data from the Open-Meteo archive API for New York City (Central Park coordinates) for 2020–2025 and saves it locally

### 3. Process CitiBike Data

- Open and run `bike_data_processing.ipynb`
- This reads all monthly CSV files from the `bike_data/` directory, aggregates them into hourly ride counts, and outputs `citibike_hourly_summary_2020_2023.csv`

### 4. Merge Datasets

- The merged CitiBike + weather dataset should be placed at `final_data/citibike_weather_merged_2020_to_2025.csv`
- This is the input file expected by the main analysis notebook

### 5. Run the Main Analysis & Forecasting

- Open and run `informatics_capstone_projectFinal_SatvikReddyKonda.ipynb`
- This notebook walks through the full pipeline:
  1. Loading and viewing data
  2. Exploratory analysis (demand trends, weekday vs weekend, seasonal plots)
  3. STL decomposition and anomaly detection
  4. Weather correlation analysis
  5. ACF/PACF analysis
  6. All forecasting models (sections 10–29)
  7. Performance summary and hyperparameter optimization
