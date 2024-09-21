"For Time Series and Visualization: Libraries"
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"Load the Dataset"
df_weather = pd.read_csv("GlobalWeatherRepository.csv")
weather_head = (df_weather.head())
weather_tail = (df_weather.tail())
weather_info = (df_weather.info())
weather_describe = (df_weather.describe())

"Checking for Missing values and Outliers"
df_weather_missval = df_weather.isnull().sum()
"Outliers--valid data point and correct or remove anomalies"
wind_mph = df_weather['wind_mph'].quantile(0.25)
pressure_mb = df_weather['pressure_mb'].quantile(0.50)
humidity = df_weather['humidity'].quantile(0.75)

'IQR(Inter-Quantile Range) = Middle Range'
IQR = humidity-wind_mph
lower_bound = wind_mph-1.5*IQR
upper_bound = humidity+1.5*IQR


'Outlier-visualize'
plt.boxplot(df_weather['wind_mph'],vert=False)
plt.title("outlier-wind_mph ")
plt.tight_layout()
plt.show()

plt.boxplot(df_weather['humidity'],vert=False)
plt.title("outlier-humidity")
plt.tight_layout()
plt.show()

"EDA--Exploratory Data Analysis"
'Hottest country'
Hottest_country = df_weather[['country','location_name','temperature_celsius','last_updated']][df_weather['temperature_celsius']==df_weather.temperature_celsius.max()]
'coolest country'
coolest_country = df_weather[['country','location_name','temperature_celsius','last_updated']][df_weather['temperature_celsius']==df_weather.temperature_celsius.min()]
'Humid country'
Humid_country = df_weather[['country','location_name','humidity','last_updated']][df_weather['humidity']==df_weather.humidity.max()].sort_values('last_updated',ascending=False).head(1)


"Extracting DateTime Index"
df_weather['last_updated_date'] = pd.to_datetime(df_weather['last_updated']).dt.date
df_weather['last_updated_time'] = pd.to_datetime(df_weather['last_updated']).dt.time
df_weather['Date'] = pd.to_datetime(df_weather['last_updated'])
df_weather['time'] = pd.to_datetime(df_weather['last_updated'])
df_weather.set_index('Date', inplace=True)
df_weather.set_index('time', inplace=True)


result = adfuller(df_weather['temperature_celsius'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


result = adfuller(df_weather['air_quality_us-epa-index'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

'visualizing Air Quality Index'
air_quality_india = df_weather[['country','air_quality_us-epa-index','last_updated_date']][df_weather['country']=='India'].set_index('last_updated_date')
air_quality_Germany = df_weather[['country','air_quality_us-epa-index','last_updated_date']][df_weather['country']=='Germany'].set_index('last_updated_date')
INDvsGER = air_quality_india.join(air_quality_Germany,lsuffix='IND',rsuffix='GER')
plt.figure(figsize=(10,8))
plt.plot(INDvsGER.index,INDvsGER['air_quality_us-epa-indexIND'])
plt.plot(INDvsGER.index,INDvsGER['air_quality_us-epa-indexGER'])
plt.xlabel('Day')
plt.ylabel('Air Quality EPA INDEX')
plt.title('India VS Germany AQI')
plt.legend()
plt.show()

"visualizing the time series-Temperature celsius with latitude"
df_weather['latitude'].plot(figsize=(10, 8), title='weather_temperature', ylabel='temperature_celsius')
plt.show()

"Decompose the time series > trend >seasonality >residual components"
decomposition = seasonal_decompose(df_weather['temperature_celsius'], model='additive', period=12)
decomposition.plot()
plt.show()


"visualizing the time series- Air_quality_us-epa-index with visibility_km "
df_weather['visibility_km'].plot(figsize=(10, 8), title='Air Quality_Index', ylabel='air_quality_us-epa-index')
plt.show()

"Decompose the time series > trend >seasonality >residual components"
decomposition = seasonal_decompose(df_weather['air_quality_us-epa-index'], model='additive', period=12)
decomposition.plot()
plt.show()

"Convert 'last_updated' to datetime and extract the year"
df_weather['last_updated_year'] = pd.to_datetime(df_weather['last_updated'])

"Group by year to get the maximum air quality index and temperature"
max_values_by_year = df_weather.groupby('last_updated').agg({'air_quality_us-epa-index': 'max','temperature_celsius':'max'}).reset_index()

"Convert the 'last_updated' column to datetime"
max_values_by_year['last_updated'] = pd.to_datetime(max_values_by_year['last_updated'])

"Extract the year from the 'last_updated' column"
max_values_by_year['year_last_updated'] = max_values_by_year['last_updated'].dt.year
max_values_df_weather = max_values_by_year[max_values_by_year['year_last_updated'] <= 2024]


"Setting 'year' as the index"
max_values_df_weather.set_index('last_updated', inplace=True)

"Using ARIMA Model - Forecasting for Air Quality Index (EPA)"
model_aqi = ARIMA(max_values_df_weather['air_quality_us-epa-index'], order=(0,1,1))
fit_aqi = model_aqi.fit()
print(fit_aqi)

"Forecasting aqi for 2025"
forecast_aqi_2025 = fit_aqi.forecast(steps=10)
print(forecast_aqi_2025)

"Using ARIMA Model - Forecasting for Temperature"
model_temp = ARIMA(max_values_df_weather['temperature_celsius'], order=(0,1,1))
fit_temp_2025 = model_temp.fit()
print(fit_temp_2025)

"Forecasting temp for 2025"
forecast_temp_2025 = fit_temp_2025.forecast(steps=10)
print(forecast_temp_2025)

"Extraction of the forecasted AQI and temperature values-series to scalar"
forecast_aqi_2025_value = forecast_aqi_2025.values[0]
forecast_temp_2025_value = forecast_temp_2025.values[0]
print(f"Forecasted Maximum Air Quality Index (EPA) for 2025: {forecast_aqi_2025_value}")
print(f"Forecasted Maximum Temperature for 2025: {forecast_temp_2025_value}°C")

"Visualizing Forecasted AQI for 2025 "
max_values_df_weather['air_quality_us-epa-index'].plot(label='Historical AQI', figsize=(12, 6))
plt.axhline(y=forecast_aqi_2025_value, color='r', linestyle='--', label='Forecasted AQI (2025)')
plt.title('Historical and Forecasted Air Quality Index (AQI)')
plt.xlabel('Year')
plt.ylabel('Air Quality Index (EPA)')
plt.legend()
plt.show()

"Visualizing Forecasted Temperature for 2025"
max_values_df_weather['temperature_celsius'].plot(label='Historical Temperature', figsize=(12, 6))
plt.axhline(y=forecast_temp_2025_value, color='r', linestyle='--', label='Forecasted Temp (2025)')
plt.title('Historical and Forecasted Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()















