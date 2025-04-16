import pandas as pd

path1='/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/original/summerOly_athletes.csv'
data = pd.read_csv(path1)
# Step 1: Identify and extract athletes whose teams have a "/" in their name
athletes_with_slash = data[data['Team'].str.contains("/", na=False)]
athlete_names_with_slash = athletes_with_slash['Name'].tolist()
total_athletes_with_slash = len(athlete_names_with_slash)

# Step 2: Analyze their medal distribution
medal_distribution = athletes_with_slash['Medal'].value_counts()

# Step 3: Remove these athletes from the dataset
data_cleaned = data[~data['Team'].str.contains("/", na=False)]

# Prepare results for display
results = {
    "Total Athletes with '/' in Team Name": total_athletes_with_slash,
    "Athlete Names (First 10)": athlete_names_with_slash[:10],
    "Medal Distribution": medal_distribution.to_dict(),
}

print(results)

# To calculate the average number of times each athlete participated:
# Check if the dataset has columns indicating participation or duplicates for each athlete.
# If not, assume each row corresponds to one participation.
#1
# Group by athlete name to count their participations
participation_counts = data_cleaned.groupby('Name').size()

# Calculate the average number of participations
average_participation = participation_counts.mean()

print(average_participation)


participation_years = data.groupby('Name')['Year'].nunique()
#2
 # Calculate the average number of distinct participation years
average_participation_years = participation_years.mean()
print("average_participation_years: ",average_participation_years)
#3
# Group by 'Sport' and 'Name', and count unique years for each athlete in each sport
sport_participation_years = data_cleaned.groupby(['Sport', 'Name'])['Year'].nunique()

# Calculate the average distinct participation years for each sport
average_years_by_sport = sport_participation_years.groupby('Sport').mean()

# Separate sports with averages higher than 2 and those very close to 1
higher_than_2 = average_years_by_sport[average_years_by_sport > 2]
close_to_1 = average_years_by_sport[(average_years_by_sport > 0.95) & (average_years_by_sport < 1.05)]

# Convert results to DataFrames for display
higher_than_2_df = higher_than_2.reset_index(name='Average Distinct Years')
close_to_1_df = close_to_1.reset_index(name='Average Distinct Years')



# Save the two dataframes as Excel files
higher_than_2_file_path = "/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/Analysis/results/higher_than_2_sports.xlsx"
close_to_1_file_path = "/Users/davidzhu/Desktop/MCM/2025/2025_Problem_C_Data/Analysis/results/close_to_1_sports.xlsx"

higher_than_2_df.to_excel(higher_than_2_file_path, index=False)
close_to_1_df.to_excel(close_to_1_file_path, index=False)

(higher_than_2_file_path, close_to_1_file_path)


import matplotlib.pyplot as plt

# Count the number of athletes for each distinct number of participation years
years_distribution = participation_years.value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(years_distribution.index, years_distribution.values, width=0.8, edgecolor='black')
plt.xlabel('Number of Distinct Participation Years', fontsize=12)
plt.ylabel('Number of Athletes', fontsize=12)
plt.title('Distribution of Athletes by Number of Distinct Participation Years', fontsize=14)
plt.xticks(years_distribution.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

#plt.show()

# Calculate the total number of athletes
total_athletes = participation_years.size

# Calculate the percentages of athletes participating in 1, 2, and 3 distinct years
percentages = {
    "1 Year": (participation_years[participation_years == 1].size / total_athletes) * 100,
    "2 Years": (participation_years[participation_years == 2].size / total_athletes) * 100,
    "3 Years": (participation_years[participation_years == 3].size / total_athletes) * 100,
}

#print(percentages)
# Filter athletes who participated in 2024
athletes_in_2024 = data_cleaned[data_cleaned['Year'] == 2024]

# Determine old and young athletes
old_athletes = athletes_in_2024[athletes_in_2024['Name'].isin(participation_years[participation_years > 1].index)]
young_athletes = athletes_in_2024[~athletes_in_2024['Name'].isin(participation_years[participation_years > 1].index)]

# Calculate percentages of medal winners and gold winners for old athletes
old_medal_winners = old_athletes[old_athletes['Medal'] != 'No medal']
old_gold_winners = old_athletes[old_athletes['Medal'] == 'Gold']
old_medal_percentage = (len(old_medal_winners) / len(old_athletes)) * 100 if len(old_athletes) > 0 else 0
old_gold_percentage = (len(old_gold_winners) / len(old_athletes)) * 100 if len(old_athletes) > 0 else 0

# Calculate percentages of medal winners and gold winners for young athletes
young_medal_winners = young_athletes[young_athletes['Medal'] != 'No medal']
young_gold_winners = young_athletes[young_athletes['Medal'] == 'Gold']
young_medal_percentage = (len(young_medal_winners) / len(young_athletes)) * 100 if len(young_athletes) > 0 else 0
young_gold_percentage = (len(young_gold_winners) / len(young_athletes)) * 100 if len(young_athletes) > 0 else 0

{
    "Old Athletes": {
        "Medal Winners Percentage": old_medal_percentage,
        "Gold Winners Percentage": old_gold_percentage,
    },
    "Young Athletes": {
        "Medal Winners Percentage": young_medal_percentage,
        "Gold Winners Percentage": young_gold_percentage,
    },
}

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Standardize and merge team names
data_cleaned['Team'] = data_cleaned['Team'].str.lower().str.strip()

# Step 2: Calculate the percentage of medal-winning and gold-winning events for each team
team_stats = data_cleaned.groupby('Team').agg(
    total_events=('Event', 'count'),
    total_medals=('Medal', lambda x: (x != 'No medal').sum()),
    total_golds=('Medal', lambda x: (x == 'Gold').sum())
)
team_stats['medal_percentage'] = (team_stats['total_medals'] / team_stats['total_events']) * 100
team_stats['gold_percentage'] = (team_stats['total_golds'] / team_stats['total_events']) * 100

# Step 3: Distribution plot of the medal-winning percentage
plt.figure(figsize=(10, 6))
plt.hist(team_stats['medal_percentage'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Medal-Winning Percentage', fontsize=12)
plt.ylabel('Number of Teams', fontsize=12)
plt.title('Distribution of Medal-Winning Percentage Across Teams', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Step 4: Find the 20 teams that won the most medals in 2024
teams_2024 = data_cleaned[data_cleaned['Year'] == 2024]
top_20_teams = teams_2024.groupby('Team').agg(total_medals=('Medal', lambda x: (x != 'No medal').sum())).nlargest(20, 'total_medals')

# Analyze old and young athletes for the top 20 teams
top_20_team_stats = []
for team in top_20_teams.index:
    team_data = teams_2024[teams_2024['Team'] == team]
    old_team_athletes = team_data[team_data['Name'].isin(participation_years[participation_years > 1].index)]
    young_team_athletes = team_data[~team_data['Name'].isin(participation_years[participation_years > 1].index)]

    old_medal_winners = old_team_athletes[old_team_athletes['Medal'] != 'No medal']
    old_gold_winners = old_team_athletes[old_team_athletes['Medal'] == 'Gold']
    young_medal_winners = young_team_athletes[young_team_athletes['Medal'] != 'No medal']
    young_gold_winners = young_team_athletes[young_team_athletes['Medal'] == 'Gold']

    top_20_team_stats.append({
        'Team': team,
        'Old Athletes Percentage': (len(old_team_athletes) / len(team_data)) * 100 if len(team_data) > 0 else 0,
        'Young Athletes Percentage': (len(young_team_athletes) / len(team_data)) * 100 if len(team_data) > 0 else 0,
        'Old Medal Winners Percentage': (len(old_medal_winners) / len(old_team_athletes)) * 100 if len(old_team_athletes) > 0 else 0,
        'Old Gold Winners Percentage': (len(old_gold_winners) / len(old_team_athletes)) * 100 if len(old_team_athletes) > 0 else 0,
        'Young Medal Winners Percentage': (len(young_medal_winners) / len(young_team_athletes)) * 100 if len(young_team_athletes) > 0 else 0,
        'Young Gold Winners Percentage': (len(young_gold_winners) / len(young_team_athletes)) * 100 if len(young_team_athletes) > 0 else 0,
    })

# Convert to DataFrame for export
top_20_team_stats_df = pd.DataFrame(top_20_team_stats)
