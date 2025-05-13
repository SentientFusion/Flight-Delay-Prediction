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
