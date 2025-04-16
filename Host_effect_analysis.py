import pandas as pd

# Load the cleaned host data from the Excel file
host_cleaned_path = '/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/original/summerOly_hosts_cleaned.xlsx'
hosts_cleaned = pd.read_excel(host_cleaned_path) #host_clean is a excel file without states, only country name is recored.

# Load the data from the uploaded CSV file
file_path = '/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/original/summerOly_medal_counts.csv'
olympic_data_medals = pd.read_csv(file_path)

# Get the unique list of countries from the 'NOC' column
countries = olympic_data_medals['NOC'].unique()
countries.sort()  # Sort the countries alphabetically
#print(countries)
# Remove non-breaking space characters and other inconsistencies from the country names
olympic_data_medals['NOC'] = olympic_data_medals['NOC'].str.replace('\xa0', '')

# Now let's check the unique countries again after cleaning
unique_countries = olympic_data_medals['NOC'].unique()
unique_countries.sort()
print(unique_countries, len(unique_countries))

# Clean the host country names to ensure consistency with the NOC data

hosts_cleaned['Host'] = hosts_cleaned['Host'].str.replace(' ', '')
hosts_cleaned['Host'] = hosts_cleaned['Host'].str.replace('UnitedStates', 'United States')
hosts_cleaned['Host'] = hosts_cleaned['Host'].str.replace('UnitedKingdom', 'Great Britain')
hosts_cleaned['Host'] = hosts_cleaned['Host'].str.replace('UnitedKingdom', 'Great Britain')
# Check if all host countries are present in the unique countries array
host_countries_corrected = hosts_cleaned['Host'].apply(lambda x: x.replace(' ', '')).unique()
set(host_countries_corrected) - set(unique_countries)

# Group the data by 'NOC' and 'Year' and sum the medals for each combination
grouped_data = olympic_data_medals.groupby(['NOC', 'Year']).agg({'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum', 'Total': 'sum'}).reset_index()
# Calculate the Total_weighted column using the specified weights for Gold, Silver, and Bronze medals
grouped_data['Total_weighted'] = grouped_data['Gold'] + grouped_data['Silver'] / 2 + grouped_data['Bronze'] / 3

# Display a sample of the dataset with the new Total_weighted column
print(grouped_data[['Year', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Total_weighted']].head())

# Creating a time series data set for each country
time_series_data = {country: grouped_data[grouped_data['NOC'] == country] for country in unique_countries}
print(time_series_data['United States']  )# Show example data for the United States as a preview
import matplotlib.pyplot as plt

# Summing the Total medals for each country and selecting the top 16
top_countries = grouped_data.groupby('NOC')['Total'].sum().nlargest(16).index

# Filter data to include only the top 16 countries
top_countries_data = grouped_data[grouped_data['NOC'].isin(top_countries)]


# Save the figure to a file

#plt.show()

from sklearn.linear_model import LinearRegression

# Function to apply linear regression for predicting medals
def predict_medals_for_host_years(data, medal_type):
    predictions = {}

    for noc in top_countries:
        # Filter data for the current NOC excluding host years
        host_years = hosts_cleaned[hosts_cleaned['Host'] == noc]['Year']
        training_data = data[(data['NOC'] == noc) & (~data['Year'].isin(host_years))]

        # Prepare training data
        X_train = training_data[['Year']]
        y_train = training_data[medal_type]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prepare data for predictions (only host years)
        X_predict = hosts_cleaned[hosts_cleaned['Host'] == noc][['Year']]
        if not X_predict.empty:
            y_pred = model.predict(X_predict)
            for year, pred in zip(X_predict['Year'], y_pred):
                predictions[(noc, year)] = pred

    return predictions

# Predicting Total and Total_weighted medals for host years
total_medal_predictions = predict_medals_for_host_years(grouped_data, 'Total')
total_weighted_medal_predictions = predict_medals_for_host_years(grouped_data, 'Total_weighted')

# Creating a DataFrame for predictions
predictions_df = pd.DataFrame(list(total_medal_predictions.items()), columns=['Country_Year', 'Total_Predicted'])
predictions_df[['NOC', 'Year']] = pd.DataFrame(predictions_df['Country_Year'].tolist(), index=predictions_df.index)
predictions_df['Weighted_Total_Predicted'] = predictions_df['Country_Year'].map(total_weighted_medal_predictions)

# Merging predictions with host data for comparison
prediction_results = predictions_df.merge(hosts_cleaned, left_on=['NOC', 'Year'], right_on=['Host', 'Year'])
#print(prediction_results[['Year', 'Host', 'Total_Predicted', 'Weighted_Total_Predicted']])
# Merging the predictions with actual medal data from the host years
host_medal_data = grouped_data[grouped_data['Year'].isin(prediction_results['Year']) & grouped_data['NOC'].isin(prediction_results['Host'])]
prediction_results = prediction_results.merge(host_medal_data[['Year', 'NOC', 'Total', 'Total_weighted']], left_on=['Year', 'Host'], right_on=['Year', 'NOC'], how='left')

# Calculating the differences
prediction_results['Difference_Total'] = prediction_results['Total'] - prediction_results['Total_Predicted']
prediction_results['Difference_Weighted_Total'] = prediction_results['Total_weighted'] - prediction_results['Weighted_Total_Predicted']

# Display the updated dataframe with differences
print(prediction_results[['Year', 'Host', 'Total_Predicted', 'Total', 'Difference_Total', 'Weighted_Total_Predicted', 'Total_weighted', 'Difference_Weighted_Total']])
