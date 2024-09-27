#Chiraath 104834008
import pandas as pd  # Used for data manipulation and analysis

# Step 1: Load the dataset
# This assumes you have already downloaded the air quality dataset and it's in CSV format
# Replace 'air_quality_data.csv' with the actual path to your dataset
air_quality_data = pd.read_csv('air_quality_data.csv')

# Step 2: Inspect the dataset
# This gives a preview of the dataset's first five rows to understand its structure
print("Preview of the original dataset:")
print(air_quality_data.head())

# Step 3: Data Cleaning
# We're dropping columns that are not necessary for analysis. In this case, 'City' and 'Date'.
# We are focusing on pollutant values, so location and time might not be relevant if we're aiming for a broader model
# .dropna() removes any rows where there are missing (NaN) values, as missing data can skew the results
air_quality_data_cleaned = air_quality_data.drop(columns=['City', 'Date']).dropna()

# Step 4: Preview the cleaned dataset
# This gives a preview of the cleaned data, after removing unnecessary columns and missing values
print("Preview of the cleaned dataset:")
print(air_quality_data_cleaned.head())

# Step 5: Focus on Key Pollutants
# We're only interested in the key pollutants identified by research: PM2.5, PM10, SO2, and O3.
# These are extracted into a new dataframe for further analysis.
relevant_pollutants = ['PM2.5', 'PM10', 'SO2', 'O3']
pollutant_data = air_quality_data_cleaned[relevant_pollutants]

# Step 6: Summary statistics of the relevant pollutant data
# This provides descriptive statistics like mean, standard deviation, min, and max for the pollutant data
# Helps to understand the range and distribution of the pollutants in the dataset
print("Summary of the relevant pollutant data:")
print(pollutant_data.describe())

# Step 7: (Optional) Save the cleaned dataset for further analysis
# This saves the cleaned and filtered dataset into a new CSV file for further use
pollutant_data.to_csv('cleaned_air_quality_data.csv', index=False)

# The cleaned dataset is now ready for further use, such as training machine learning models
