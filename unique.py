import pandas as pd

# Read the dataset
data = pd.read_csv("output.csv")

# Extract unique airport codes from the 'dest' column
unique_airport_codes = data['dest'].unique()

# Print the unique airport codes
print(unique_airport_codes)
