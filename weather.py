import requests
from fuzzywuzzy import process

unique_weather_conditions = [
    'Partly Cloudy', 'Mostly Cloudy', 'Fair', 'Cloudy', 'Light Snow', 'Light Snow / Windy', 
    'Snow and Sleet', 'Wintry Mix', 'Rain', 'Light Rain', 'Heavy Rain', 'Light Drizzle', 
    'Fog', 'Cloudy / Windy', 'Partly Cloudy / Windy', 'Fair / Windy', 'Light Sleet', 
    'Light Freezing Drizzle', 'Snow / Freezing Rain', 'Light Snow / Freezing Rain', 'Snow', 
    'Mostly Cloudy / Windy', 'Light Rain / Windy', 'Rain and Snow', 'Snow / Windy', 
    'Heavy Snow / Windy', 'Heavy Snow', 'Blowing Snow / Windy', 'Haze', 'Unknown Precipitation', 
    'Blowing Snow', 'Rain and Sleet', 'Rain / Freezing Rain', 'Wintry Mix / Windy', 'T-Storm', 
    'Light Snow and Sleet', 'Light Drizzle / Windy', 'Thunder in the Vicinity', 'Drizzle and Fog', 
    'Patches of Fog', 'Heavy Rain / Windy', 'Rain / Windy', 'Squalls', 'Light Rain with Thunder'
]

def fetch_weather_forecast(api_key, city_name, date):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch weather forecast data for {city_name}.")
        return None

def filter_forecast_at_1500(weather_forecast):
    forecast_at_1500 = []
    for forecast in weather_forecast['list']:
        if forecast['dt_txt'].split()[1] == '15:00:00':
            forecast_at_1500.append(forecast)
    return forecast_at_1500

def map_weather_condition(weather_condition):
    best_match = process.extractOne(weather_condition, unique_weather_conditions)[0]
    return best_match
