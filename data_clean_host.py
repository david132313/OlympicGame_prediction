import pandas as pd

# Load the cleaned host data from the Excel file
host_cleaned_path = '/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/summerOly_hosts_cleaned.xlsx'
hosts_cleaned = pd.read_excel(host_cleaned_path) #host_clean is a excel file without states, only country name is recored.

# Load the data from the uploaded CSV file
file_path = '/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/original/summerOly_medal_counts.csv'
olympic_data_medals = pd.read_csv(file_path)


# Pivot the data to make 'Year' as columns and 'NOC' as rows for the 'Total' medals column
medals_time_series = olympic_data_medals.pivot_table(index='NOC', columns='Year', values='Total', fill_value=0)
medals_time_series_transposed = medals_time_series.T
# 移除所有字符串字段中的空格
medals_time_series_transposed = medals_time_series_transposed.map(lambda x: x.strip() if isinstance(x, str) else x)
# Display the transformed data
#print(medals_time_series_transposed.head(), medals_time_series_transposed.shape)

import os

directory = '/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data'
if not os.path.exists(directory):
    os.makedirs(directory)

path = os.path.join(directory, 'time_series_medal.csv')
medals_time_series_transposed.to_csv(path, index=True)
 # Set index=False if you don't want to save the DataFrame index as a column in the CSV

#load the time series data set 
time_series_data_path='/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/time_series_medal.csv'
time_series_data = pd.read_csv(time_series_data_path)

#merge time series medal dat set with host data set
# Create a dictionary from the hosts data for easy lookup
host_dict = hosts_cleaned.set_index('Year')['Host'].to_dict()

# Add a new column for each country in the medals data to include the binary host indicator

for country in time_series_data.columns[1:]:  # skip 'Year'
    time_series_data[country] = time_series_data.apply(
        lambda row: (row[country], 1 if host_dict.get(row['Year'], None) == country.strip() else 0), axis=1)


# Display the modified medals data to confirm the structure
#print(time_series_data.head())


if not os.path.exists(directory):
    os.makedirs(directory)

path = os.path.join(directory, 'time_series_medal_merge_hosts.csv')
time_series_data.to_csv(path, index=True)