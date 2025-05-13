from flask import Flask, render_template, request
import pandas as pd
from prediction import preprocess_data_rf, preprocess_data_xgb, train_random_forest, train_xgboost, evaluate_model
from weather import fetch_weather_forecast, filter_forecast_at_1500, map_weather_condition
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Read the dataset
data = pd.read_csv("output.csv")

# Extract unique airports
unique_departure_airports = data['origin'].unique()
unique_arrival_airports = data['dest'].unique()

# Map airport codes to city names
city_map = {
    'IAH': 'Houston',
    'MIA': 'Miami',
    'BQN': 'Aguadilla',
    'ATL': 'Atlanta',
    'ORD': 'Chicago',
    'FLL': 'Fort Lauderdale',
    'IAD': 'Washington D.C.',
    'MCO': 'Orlando',
    'PBI': 'West Palm Beach',
    'TPA': 'Tampa',
    'LAX': 'Los Angeles',
    'SFO': 'San Francisco',
    'LAS': 'Las Vegas',
    'MSP': 'Minneapolis',
    'DTW': 'Detroit',
    'RSW': 'Fort Myers',
    'SJU': 'San Juan',
    'PHX': 'Phoenix',
    'BWI': 'Baltimore',
    'CLT': 'Charlotte',
    'DFW': 'Dallas',
    'BOS': 'Boston',
    'BUF': 'Buffalo',
    'DEN': 'Denver',
    'SNA': 'Santa Ana',
    'MSY': 'New Orleans',
    'SLC': 'Salt Lake City',
    'XNA': 'Bentonville',
    'MKE': 'Milwaukee',
    'SEA': 'Seattle',
    'ROC': 'Rochester',
    'SYR': 'Syracuse',
    'SRQ': 'Sarasota',
    'RDU': 'Raleigh-Durham',
    'CMH': 'Columbus',
    'JAX': 'Jacksonville',
    'CHS': 'Charleston',
    'MEM': 'Memphis',
    'PIT': 'Pittsburgh',
    'SAN': 'San Diego',
    'DCA': 'Washington D.C.',
    'CLE': 'Cleveland',
    'STL': 'St. Louis',
    'MYR': 'Myrtle Beach',
    'JAC': 'Jackson Hole',
    'MDW': 'Chicago (Midway)',
    'HNL': 'Honolulu',
    'BNA': 'Nashville',
    'AUS': 'Austin',
    'BTV': 'Burlington',
    'PHL': 'Philadelphia',
    'STT': 'St. Thomas',
    'EGE': 'Eagle',
    'AVL': 'Asheville',
    'PWM': 'Portland (Maine)',
    'IND': 'Indianapolis',
    'SAV': 'Savannah',
    'CAK': 'Akron',
    'HOU': 'Houston (Hobby)',
    'LGB': 'Long Beach',
    'DAY': 'Dayton',
    'ALB': 'Albany',
    'BDL': 'Hartford',
    'MHT': 'Manchester',
    'MSN': 'Madison',
    'GSO': 'Greensboro',
    'CVG': 'Cincinnati',
    'BUR': 'Burbank',
    'RIC': 'Richmond',
    'GSP': 'Greenville',
    'GRR': 'Grand Rapids',
    'MCI': 'Kansas City',
    'ORF': 'Norfolk',
    'SAT': 'San Antonio',
    'SDF': 'Louisville',
    'PDX': 'Portland',
    'SJC': 'San Jose',
    'OMA': 'Omaha',
    'CRW': 'Charleston (West Virginia)',
    'OAK': 'Oakland',
    'SMF': 'Sacramento',
    'TYS': 'Knoxville',
    'PVD': 'Providence',
    'DSM': 'Des Moines',
    'PSE': 'Ponce',
    'TUL': 'Tulsa',
    'BHM': 'Birmingham',
    'OKC': 'Oklahoma City',
    'CAE': 'Columbia',
    'HDN': 'Steamboat Springs',
    'BZN': 'Bozeman',
    'MTJ': 'Montrose',
    'EYW': 'Key West',
    'PSP': 'Palm Springs',
    'ACK': 'Nantucket',
    'BGR': 'Bangor',
    'ABQ': 'Albuquerque',
    'ILM': 'Wilmington',
    'MVY': 'Martha\'s Vineyard',
    'SBN': 'South Bend',
    'LEX': 'Lexington',
    'CHO': 'Charlottesville'
}

# Extract unique dates within the next 5 days
dates = pd.date_range(start=pd.Timestamp.now(), periods=5).strftime('%Y-%m-%d').tolist()

# Preprocess data for RandomForestClassifier
data_rf = preprocess_data_rf(data.copy())

# Set train-test split ratio and random state
test_size = 0.2
random_state = 42

# Split the dataset into features (X) and target (y) for RandomForestClassifier
X_rf = data_rf.drop(columns=["delay_minutes"])
y_rf = data_rf["delay_minutes"] > 0  # 1 if delayed, 0 if not delayed
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=test_size, random_state=random_state)

# Train RandomForestClassifier model
classifier_rf = train_random_forest(X_train_rf, y_train_rf)
evaluate_model(classifier_rf, X_test_rf, y_test_rf, "Random Forest Classifier")

# Preprocess data for XGBoostRegressor
data_xgb = preprocess_data_xgb(data.copy())
categorical_features = ["carrier", "origin", "dest", "name", "flight_status", "Rain", "Weather_Condition"]
numerical_features = ['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'air_time', 'distance', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']

# Split the dataset into features (X) and target (y) for XGBoostRegressor
X_xgb = data_xgb.drop(columns=["delay_minutes"])
y_xgb = data_xgb["delay_minutes"]
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=test_size, random_state=random_state)

# Train XGBoostRegressor model
model_xgb = train_xgboost(X_train_xgb, y_train_xgb, categorical_features, numerical_features)
evaluate_model(model_xgb, X_test_xgb, y_test_xgb, "XGBoost Regressor")

@app.route('/')
def index():
    return render_template('index.html', departure_airports=unique_departure_airports, arrival_airports=unique_arrival_airports, dates=dates)

@app.route('/flights', methods=['POST'])
def flights():
    departure = request.form['departure']
    arrival = request.form['arrival']
    date = request.form['date']
    
    # Filter flights for the given departure, arrival, and date
    available_flights = data[(data['origin'] == departure) & (data['dest'] == arrival)]
    
    # Randomly select up to 10 flights, minimum of 5
    available_flights = available_flights.sample(n=min(10, len(available_flights)), random_state=42)
    
    # Ensure there are at least 5 flights
    if len(available_flights) < 5:
        available_flights = data.sample(n=5, random_state=42)
    
    flights = available_flights.to_dict('records')
    
    return render_template('flights.html', flights=flights, departure=departure, arrival=arrival, date=date)

@app.route('/delay', methods=['POST'])
def delay():
    flight_id = request.form['flight_id']
    departure = request.form['departure']
    arrival = request.form['arrival']
    date = request.form['date']
    
    # Get the selected flight data
    selected_flight = data[data['flight'] == int(flight_id)].iloc[0]
    
    # Fetch weather data using city name
    api_key = 'Your_API_Key'
    weather_forecast = fetch_weather_forecast(api_key, city_map[arrival], date)
    if weather_forecast:
        forecast_at_1500 = filter_forecast_at_1500(weather_forecast)
        if forecast_at_1500:
            weather_condition = forecast_at_1500[0]['weather'][0]['description']
            mapped_weather_condition = map_weather_condition(weather_condition)
            selected_flight['Weather_Condition'] = mapped_weather_condition
    
    # Preprocess the flight data for RandomForestClassifier
    sample_tuple_rf = preprocess_data_rf(pd.DataFrame([selected_flight]))
    
    # Make predictions using the RandomForestClassifier model
    prediction_rf = classifier_rf.predict(sample_tuple_rf.drop(columns=["delay_minutes"]))
    delay_status = "Delayed" if prediction_rf[0] else "Not Delayed"
    
    # Calculate the predicted delay time using XGBoostRegressor
    if delay_status == "Delayed":
        # Preprocess the flight data for XGBoostRegressor
        sample_tuple_xgb = preprocess_data_xgb(pd.DataFrame([selected_flight]))
        
        # Make predictions using the XGBoostRegressor model
        prediction_xgb = model_xgb.predict(sample_tuple_xgb.drop(columns=["delay_minutes"]))
        
        # Predicted delay time directly
        predicted_delay_time = prediction_xgb[0]
    else:
        predicted_delay_time = 0
    
    return render_template('delay.html', delay_status=delay_status, predicted_delay_time=predicted_delay_time)

if __name__ == '__main__':
    app.run(debug=True)
