import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load data (CSV file name unga file name ah maathunga)
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

df = pd.read_csv(url)
df = df[['location', 'date', 'total_cases', 'total_deaths']]

# 2. Clean data
countries = ['Brazil', 'India', 'United States']
df = df[df['location'].isin(countries)]
df['date'] = pd.to_datetime(df['date'])
df = df.dropna().sort_values(['location', 'date'])
print("Shape after cleaning:", df.shape)

# 3. Line chart: total cases growth
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='total_cases', hue='location')
plt.title('COVID-19 Total Cases Growth')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.savefig('covid_cases.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Heatmap: monthly new cases
df['new_cases'] = df.groupby('location')['total_cases'].diff()
df['month'] = df['date'].dt.to_period('M').astype(str)
pivot = df.pivot_table(index='location', columns='month',
                       values='new_cases', aggfunc='sum')

plt.figure(figsize=(16, 4))
sns.heatmap(pivot, cmap='YlOrRd')
plt.title('Monthly New Cases Heatmap')
plt.xlabel('Month')
plt.ylabel('Country')
plt.xticks(rotation=90)
plt.savefig('covid_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Prediction chart: India, next 30 days (simple trend line)
india = df[df['location'] == 'India'].tail(90)
x = np.arange(len(india))
y = india['total_cases'].values
coef = np.polyfit(x, y, 1)

future_x = np.arange(len(india), len(india) + 30)
future_pred = np.polyval(coef, future_x)
future_dates = pd.date_range(india['date'].iloc[-1] + pd.Timedelta(days=1),
                             periods=30)

plt.figure(figsize=(12, 6))
plt.plot(india['date'], y, label='Actual')
plt.plot(future_dates, future_pred, linestyle='--', label='Predicted')
plt.title('India: Total Cases Prediction (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('covid_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
# 6. Deaths chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='total_deaths', hue='location')
plt.title('COVID-19 Total Deaths')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.xticks(rotation=45)
plt.savefig('covid_deaths.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Death rate chart
latest = df.groupby('location').last().reset_index()
latest['death_rate'] = latest['total_deaths'] / latest['total_cases'] * 100

plt.figure(figsize=(8, 5))
sns.barplot(data=latest, x='location', y='death_rate')
plt.title('Death Rate (%) by Country')
plt.xlabel('Country')
plt.ylabel('Death Rate (%)')
plt.savefig('covid_death_rate.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nCOVID-19 Data Analysis Completed!")
