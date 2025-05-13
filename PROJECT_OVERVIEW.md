# 🚧 Project Overview: Flight Delay
_Generated on 2025-05-13T17:04:25.541008_

## 📂 Folder Structure
```
├── .idea/
│   ├── .gitignore
│   ├── Flight Delay.iml
│   ├── misc.xml
│   ├── modules.xml
│   └── workspace.xml
├── app.py
├── output.csv
├── prediction.py
├── PROJECT_PRESENTATION-1_[Autosaved][1].pptx
├── static/
│   ├── back.jpg
│   └── style.css
├── templates/
│   ├── delay.html
│   ├── flights.html
│   └── index.html
├── test.csv
├── unique.py
└── weather.py
```

## 📄 Code Files

### `app.py`
- **Lines:** 222
- **Last Modified:** 2024-07-09T10:59:16.429543

```
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
    api_key = '5e9f4cb05bc73d1d5dc5f9ff51667a6b'
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

```

### `prediction.py`
- **Lines:** 174
- **Last Modified:** 2024-07-09T01:14:38.974838

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import weather

def preprocess_data_rf(data):
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    data[['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'air_time', 'distance', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] = imputer.fit_transform(data[['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'air_time', 'distance', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']])
    imputer = SimpleImputer(strategy="most_frequent")
    data[['carrier', 'flight', 'origin', 'dest', 'name', 'flight_status', 'Rain', 'Weather_Condition']] = imputer.fit_transform(data[['carrier', 'flight', 'origin', 'dest', 'name', 'flight_status', 'Rain', 'Weather_Condition']])

    # Encode categorical variables
    label_encoders = {}
    for column in ["carrier", "origin", "dest", "name", "flight_status", "Rain", "Weather_Condition"]:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    return data

def preprocess_data_xgb(data):
    # Drop rows with missing delay time values (since that's what we're trying to predict)
    data = data.dropna(subset=["delay_minutes"])

    return data

def preprocess_data_lr(data):
    # Drop rows with missing delay time values (since that's what we're trying to predict)
    data = data.dropna(subset=["delay_minutes"])

    return data

def train_random_forest(X_train, y_train):
    classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt', random_state=42)
    classifier_rf.fit(X_train, y_train)
    return classifier_rf

def train_xgboost(X_train, y_train, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', SimpleImputer(strategy="mean"), numerical_features)
        ])
    model_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', XGBRegressor(objective ='reg:squarederror'))])
    model_xgb.fit(X_train, y_train)
    return model_xgb

def train_linear_regression(X_train, y_train, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', SimpleImputer(strategy="mean"), numerical_features)
        ])
    model_lr = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())]) 
    model_lr.fit(X_train, y_train)
    return model_lr

def evaluate_model(model, X_test, y_test, model_name):
    if isinstance(model, RandomForestClassifier):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy}")
    elif isinstance(model, Pipeline):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} Mean Squared Error: {mse}")
        print(f"{model_name} R-squared: {r2}")

# Read the dataset
data = pd.read_csv("output.csv")

# Set train-test split ratio and random state
test_size = 0.2
random_state = 42

# Preprocess the data for Random Forest Classifier
data_rf = preprocess_data_rf(data.copy())

# Preprocess the data for XGBoost Regressor
data_xgb = preprocess_data_xgb(data.copy())
categorical_features = ["carrier", "origin", "dest", "name", "flight_status", "Rain", "Weather_Condition"]
numerical_features = ['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'air_time', 'distance', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']

# Preprocess the data for Linear Regression Model
data_lr = preprocess_data_lr(data.copy())

# Split the dataset into features (X) and target (y) for Random Forest Classifier
X_rf = data_rf.drop(columns=["delay_minutes"])
y_rf = data_rf["delay_minutes"] > 0  # 1 if delayed, 0 if not delayed
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=test_size, random_state=random_state)

# Split the dataset into features (X) and target (y) for XGBoost Regressor
X_xgb = data_xgb.drop(columns=["delay_minutes"])
y_xgb = data_xgb["delay_minutes"]
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=test_size, random_state=random_state)

# Split the dataset into features (X) and target (y) for Linear Regression Model
X_lr = data_lr.drop(columns=["delay_minutes"])
y_lr = data_lr["delay_minutes"]
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=test_size, random_state=random_state)

# Train and evaluate Random Forest Classifier
classifier_rf = train_random_forest(X_train_rf, y_train_rf)
evaluate_model(classifier_rf, X_test_rf, y_test_rf, "Random Forest Classifier")

# Train and evaluate XGBoost Regressor
model_xgb = train_xgboost(X_train_xgb, y_train_xgb, categorical_features, numerical_features)
evaluate_model(model_xgb, X_test_xgb, y_test_xgb, "XGBoost Regressor")

# Train and evaluate Linear Regression Model
model_lr = train_linear_regression(X_train_lr, y_train_lr, categorical_features, numerical_features)
evaluate_model(model_lr, X_test_lr, y_test_lr, "Linear Regression Model")

# Read sample tuples from test.csv
test_data = pd.read_csv("test.csv")

# Take the first two rows as sample tuples
sample_tuple_1 = test_data.iloc[0]
sample_tuple_2 = test_data.iloc[1]

# Preprocess the sample tuples for RandomForestClassifier
sample_tuple_1_rf_processed = preprocess_data_rf(sample_tuple_1.to_frame().T)
sample_tuple_2_rf_processed = preprocess_data_rf(sample_tuple_2.to_frame().T)

# Split the sample tuples into features (X) and target (y) for RandomForestClassifier
X_sample_tuple_1_rf = sample_tuple_1_rf_processed.drop(columns=["delay_minutes"])
X_sample_tuple_2_rf = sample_tuple_2_rf_processed.drop(columns=["delay_minutes"])

# Make predictions using the RandomForestClassifier model
prediction_rf_1 = classifier_rf.predict(X_sample_tuple_1_rf)
prediction_rf_2 = classifier_rf.predict(X_sample_tuple_2_rf)

# Preprocess the sample tuples for XGBoost Regressor
sample_tuple_1_xgb_processed = preprocess_data_xgb(sample_tuple_1.to_frame().T)
sample_tuple_2_xgb_processed = preprocess_data_xgb(sample_tuple_2.to_frame().T)

# Make predictions using the XGBoost Regressor model
prediction_xgb_1 = model_xgb.predict(sample_tuple_1_xgb_processed)
prediction_xgb_2 = model_xgb.predict(sample_tuple_2_xgb_processed)

# Preprocess the sample tuples for Linear Regression Model
sample_tuple_1_lr_processed = preprocess_data_lr(sample_tuple_1.to_frame().T)
sample_tuple_2_lr_processed = preprocess_data_lr(sample_tuple_2.to_frame().T)

# Make predictions using the Linear Regression Model
prediction_lr_1 = model_lr.predict(sample_tuple_1_lr_processed)
prediction_lr_2 = model_lr.predict(sample_tuple_2_lr_processed)

# Convert boolean predictions to strings
prediction_rf_1_str = "Delayed" if prediction_rf_1[0] else "Not Delayed"
prediction_rf_2_str = "Delayed" if prediction_rf_2[0] else "Not Delayed"

# Display actual and predicted values for Sample Tuple 1
print("Sample Tuple 1:")
print("Random Forest Classifier Prediction:", prediction_rf_1_str)
print("XGBoost Regressor Prediction:", prediction_xgb_1)
print("Linear Regression Model Prediction:", prediction_lr_1)
print()

# Display actual and predicted values for Sample Tuple 2
print("Sample Tuple 2:")
print("Random Forest Classifier Prediction:", prediction_rf_2_str)
print("XGBoost Regressor Prediction:", prediction_xgb_2)
print("Linear Regression Model Prediction:", prediction_lr_2)

```

### `static\style.css`
- **Lines:** 97
- **Last Modified:** 2024-07-09T10:28:40.247331

```
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-image: url("back.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    width: 100%;
    height: 100vh;
}

h1 {
    background-color: #007bff;
    color: #fff;
    padding: 10px 0;
    text-align: center;
    margin: 0;
}

form {
    max-width: 600px;
    margin: 20px auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

label {
    display: block;
    margin-bottom: 5px;
}

select, button {
    width: 100%;
    padding: 10px;
    margin-bottom: 20px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
}

ul {
    list-style-type: none;
    padding: 0;
}

li {
    background-color: #fff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

li span {
    flex-grow: 1;
    padding-right: 10px; /* Add padding to the right to create space between text and button */
}

a {
    display: block;
    text-align: center;
    padding: 10px;
    margin-top: 20px;
    background-color: #007bff;
    color: #fff;
    text-decoration: none;
    border-radius: 4px;
}

a:hover {
    background-color: #0056b3;
}

/* New styles for the output page */
.centered-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center;
}

.output-box {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    max-width: 500px;
    width: 100%;
}

```

### `templates\delay.html`
- **Lines:** 21
- **Last Modified:** 2024-07-09T10:28:24.983809

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Delay Prediction</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="centered-container">
        <h1>Flight Delay Prediction</h1>
        <div class="output-box">
            <p>The flight is <strong>{{ delay_status }}</strong>.</p>
            {% if delay_status == "Delayed" %}
            <p>Predicted delay time: <strong>{{ predicted_delay_time }} minutes</strong>.</p>
            {% endif %}
            <a href="{{ url_for('index') }}">Back to Flight Booking</a>
        </div>
    </div>
</body>
</html>

```

### `templates\flights.html`
- **Lines:** 26
- **Last Modified:** 2024-07-09T00:07:05.057177

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Available Flights</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Available Flights from {{ departure }} to {{ arrival }} on {{ date }}</h1>
    <ul>
        {% for flight in flights %}
        <li>
            <span>{{ flight['flight'] }} - {{ flight['name'] }} - {{ '{:02d}:{:02d}'.format(flight['dep_time'] // 100, flight['dep_time'] % 100) }}</span>
            <form action="{{ url_for('delay') }}" method="post">
                <input type="hidden" name="flight_id" value="{{ flight['flight'] }}">
                <input type="hidden" name="departure" value="{{ departure }}">
                <input type="hidden" name="arrival" value="{{ arrival }}">
                <input type="hidden" name="date" value="{{ date }}">
                <button type="submit">Select</button>
            </form>
        </li>
        {% endfor %}
    </ul>
</body>
</html>

```

### `templates\index.html`
- **Lines:** 33
- **Last Modified:** 2024-07-08T23:42:21.905970

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Booking</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Book Your Flight</h1>
    <form action="{{ url_for('flights') }}" method="post">
        <label for="departure">Departure Airport:</label>
        <select id="departure" name="departure">
            {% for airport in departure_airports %}
            <option value="{{ airport }}">{{ airport }}</option>
            {% endfor %}
        </select>
        <label for="arrival">Arrival Airport:</label>
        <select id="arrival" name="arrival">
            {% for airport in arrival_airports %}
            <option value="{{ airport }}">{{ airport }}</option>
            {% endfor %}
        </select>
        <label for="date">Date:</label>
        <select id="date" name="date">
            {% for date in dates %}
            <option value="{{ date }}">{{ date }}</option>
            {% endfor %}
        </select>
        <button type="submit">Check Available Flights</button>
    </form>
</body>
</html>

```

### `unique.py`
- **Lines:** 10
- **Last Modified:** 2024-07-09T08:10:28.345134

```
import pandas as pd

# Read the dataset
data = pd.read_csv("output.csv")

# Extract unique airport codes from the 'dest' column
unique_airport_codes = data['dest'].unique()

# Print the unique airport codes
print(unique_airport_codes)

```

### `weather.py`
- **Lines:** 35
- **Last Modified:** 2024-07-08T23:40:24.313951

```
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

```
