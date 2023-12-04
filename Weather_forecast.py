#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots
from xgboost import XGBRegressor, DMatrix, train as xgb_train
#%%
# Data Acquisition
# load data
file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Weather_forecast_data.csv'
data = pd.read_csv(file_path,delimiter=';')
#%%
data.head()
data.tail()
# Data cleaning
#%% 
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], utc=True)
# Sorting the DataFrame by the 'Date' column in ascending order
data = data.sort_values(by='Date', ascending=True)
#%%
# Selecting specified columns
selected_columns= ['Date', 'Vitesse du vent moyen 10 mn', 'Température (°C)', 'Humidité']
weather_data = data[selected_columns]

#%%
# Converting 'Date' to datetime format and setting it as the index
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data.set_index('Date', inplace=True)
# Displaying the first few rows of the modified DataFrame
print(weather_data.head())

#%%
# Checking for missing values
missing_values = weather_data.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

#%%
# Data visualization
# Descriptive Statistics
descriptive_stats = weather_data.describe()
print(descriptive_stats)
print("Median:\n", weather_data.median())
print("Mode:\n", weather_data.mode().iloc[0])

# Correlation Analysis with Significance
correlation_matrix = weather_data.corr()
p_values = weather_data.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlation_matrix.shape)
significant_correlation = p_values < 0.05
print(correlation_matrix)
print("Significant Correlations:\n", significant_correlation)

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Temperature Trend
axs[0].plot(weather_data.index, weather_data['Température (°C)'], color='red')
axs[0].set_title('Temperature Trend')
axs[0].set_ylabel('Temperature (°C)')

# Humidity Trend
axs[1].plot(weather_data.index, weather_data['Humidité'], color='blue')
axs[1].set_title('Humidity Trend')
axs[1].set_ylabel('Humidity (%)')

# Wind Speed Trend
axs[2].plot(weather_data.index, weather_data['Vitesse du vent moyen 10 mn'], color='green')
axs[2].set_title('Wind Speed Trend')
axs[2].set_ylabel('Wind Speed (km/h)')

plt.tight_layout()
plt.savefig(r'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/weather_trends.svg')
plt.show()

# Heatmap of Correlation Matrix with Significance
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, mask=~significant_correlation, cmap='coolwarm')
plt.title('Correlation between Weather Parameters with Significance')
plt.show()

# %%
# interactive visualization 
# Resetting the index so 'Date' becomes a column (needed for Plotly)
weather_data_reset = weather_data.reset_index()

# Creating traces for each parameter with custom colors and customized hover information
trace1 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Température (°C)'],
    mode='lines',
    name='Temperature',
    line=dict(color='red'),
    hoverinfo='x+y+name',
    hovertemplate='%{y} °C on %{x}<extra></extra>'
)

trace2 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Humidité'],
    mode='lines',
    name='Humidity',
    line=dict(color='blue'),
    hoverinfo='x+y+name',
    hovertemplate='%{y}% on %{x}<extra></extra>'
)

trace3 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Vitesse du vent moyen 10 mn'],
    mode='lines',
    name='Wind Speed',
    line=dict(color='green'),
    hoverinfo='x+y+name',
    hovertemplate='%{y} km/h on %{x}<extra></extra>'
)

# Combining traces
data = [trace1, trace2, trace3]

# Updating layout with responsive and aesthetic enhancements
layout = go.Layout(
    title='Weather Trends in Montpellier',
    title_x=0.5, # Center the title
    xaxis=dict(
        title='Date',
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(step='all', label='All')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    ),
    yaxis=dict(title='Measurements'),
    legend=dict(
        x=1.05,
        y=1,
        orientation="v"
    ),
    hovermode='closest',
    margin=dict(r=150),
    autosize=True, # Make layout responsive
    font=dict(size=12),
    paper_bgcolor="LightSteelBlue", # Change background color
)

# Creating figure
fig = go.Figure(data=data, layout=layout)

# Displaying figure
fig.show()

#%%
# Predictive Analysis
# Function to create time series features from datetime index
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Load data
file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Weather_forecast_data.csv'
data = pd.read_csv(file_path, delimiter=';')

# Data Cleaning
# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], utc=True)
data.set_index('Date', inplace=True)

# Sorting the DataFrame by the 'Date' column in ascending order
data = data.sort_values(by='Date', ascending=True)

# Select specified columns (assuming these columns exist in your data)
selected_columns = ['Vitesse du vent moyen 10 mn', 'Température (°C)', 'Humidité']
weather_data = data[selected_columns]
dates = weather_data.index
#%%
# Add time series features to the DataFrame
weather_data = create_features(weather_data)
weather_data.head()
#%%
# Prepare the dataset
X = weather_data.drop('Température (°C)', axis=1)
y = weather_data['Température (°C)']
#%% 
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%% 
# Split data into training and testing sets along with dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_scaled, y, dates, test_size=0.2, random_state=42)

# Train the XGBoost model (assuming XGBRegressor is used here)
xgb_model = XGBRegressor(n_estimators=1000, max_depth=3, eta=0.1, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Predict using the model
y_pred = xgb_model.predict(X_test)

# %%
# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
# %%
# Prepare for future prediction
future_dates = pd.date_range(start='2023-11-30', end='2023-12-30', freq='3H')
future_data = pd.DataFrame(index=future_dates)

# Add the same time-related features as in the original dataset

future_data['Vitesse du vent moyen 10 mn'] = np.random.normal(10, 2, len(future_data))
future_data['Humidité'] = np.random.randint(30, 80, len(future_data))
future_data['hour'] = future_data.index.hour
future_data['dayofweek'] = future_data.index.dayofweek
future_data['quarter'] = future_data.index.quarter
future_data['month'] = future_data.index.month
future_data['year'] = future_data.index.year
future_data['dayofyear'] = future_data.index.dayofyear
future_data['dayofmonth'] = future_data.index.day
future_data['weekofyear'] = future_data.index.isocalendar().week

#%%
# Scale the future data
future_features_scaled = scaler.transform(future_data)
#%%
# Predict future temperatures
future_temps = xgb_model.predict(future_features_scaled)
# %%
# Save and visualize the future predictions
future_temps_df = pd.DataFrame({'Date': future_dates, 'Predicted_Temperature_C': future_temps})
csv_file_path = 'predicted_temperatures.csv'
future_temps_df.to_csv(csv_file_path, index=False)

#%%
# Visualization code for future predictions
# Convert 'Date' column back to datetime for plotting
future_temps_df['Date'] = pd.to_datetime(future_temps_df['Date'])

# Visualization
plt.figure(figsize=(15, 7))
plt.plot(future_temps_df['Date'], future_temps_df['Predicted_Temperature_C'], label='Predicted Temperature', color='orange', marker='o', markersize=4)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Future Predicted Temperature from 30 Nov to 30 Dec 2023 (Every 3 Hours)')
plt.legend()
plt.grid(True)

# Format the x-axis for better readability
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  # Adjust the number of bins as needed
plt.xticks(rotation=45)

# Show the plot
plt.show()

# Output the path to the saved CSV file
print(f"Predicted temperatures saved to: {csv_file_path}")
#%%
# Sort the test data by dates to avoid crossover lines in the plot
sorted_indices = np.argsort(dates_test)
sorted_dates_test = dates_test[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# Interactive Visualization Actual vs Predicted Temperature : 

# Create a Plotly figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add trace for actual temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_test, name='Actual', mode='markers+lines', marker=dict(color='blue'), line=dict(width=2)),
    secondary_y=False,
)

# Add trace for predicted temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_pred, name='Predicted', mode='markers+lines', marker=dict(color='red'), line=dict(width=2)),
    secondary_y=False,
)

# Set figure layout
fig.update_layout(
    title_text='Actual vs Predicted Temperature',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    legend_title='Legend',
    hovermode='x unified'
)

# Update y-axes titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# Show the figure
fig.show()
# %%
# Sort the test data by dates to avoid crossover lines in the plot
sorted_indices = np.argsort(dates_test)
sorted_dates_test = dates_test[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]
# Visualizing Actual vs Predicted Temperature with improved clarity
plt.figure(figsize=(15, 7))
plt.plot(sorted_dates_test, sorted_y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5, alpha=0.7)
plt.plot(sorted_dates_test, sorted_y_pred, label='Predicted', color='red', marker='x', linestyle='-', markersize=5, alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()
#%%
# Creating a Plotly figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Adding trace for actual temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_test, name='Actual', mode='markers+lines', marker=dict(color='blue'), line=dict(width=2)),
    secondary_y=False,
)

# Adding trace for initial predicted temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_pred, name='Initial Predicted', mode='markers+lines', marker=dict(color='red'), line=dict(width=2)),
    secondary_y=False,
)

# Adding trace for future predicted temperatures
fig.add_trace(
    go.Scatter(x=future_temps_df['Date'], y=future_temps_df['Predicted_Temperature_C'], name='Future Predicted', mode='markers+lines', marker=dict(color='green'), line=dict(width=2)),
    secondary_y=False,
)

# Setting figure layout
fig.update_layout(
    title='Actual vs Predicted Temperature',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    legend_title='Legend',
    hovermode='x unified'
)

# Updating y-axis titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)

# Adding range slider and selector
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Showing the figure
fig.show()

#%%

# Convert sorted_dates_test and future_temps_df['Date'] to datetime if not already
sorted_dates_test = pd.to_datetime(sorted_dates_test)
future_temps_df['Date'] = pd.to_datetime(future_temps_df['Date'])
#%%
# Define date ranges for the plots
dec_dates = (sorted_dates_test >= '2023-11-23') & (sorted_dates_test <= '2023-11-29')
jan_dates = (sorted_dates_test >= '2023-01-01') & (sorted_dates_test <= '2023-01-31')
jul_dates = (sorted_dates_test >= '2023-07-01') & (sorted_dates_test <= '2023-07-07')

# Create a subplot for each date range
fig = make_subplots(rows=3, cols=1, subplot_titles=('Last Week of November', 'First Month of January', 'First Week of July'))

# Function to add traces to a subplot with different colors and legend names
def add_traces(row, dates_range, actual_color, predicted_color, actual_legend, predicted_legend):
    # Actual Temperature
    fig.add_trace(
        go.Scatter(x=sorted_dates_test[dates_range], y=sorted_y_test[dates_range], name=actual_legend, mode='lines', line=dict(color=actual_color)),
        row=row, col=1
    )
    # Predicted Temperature
    fig.add_trace(
        go.Scatter(x=sorted_dates_test[dates_range], y=sorted_y_pred[dates_range], name=predicted_legend, mode='lines', line=dict(color=predicted_color)),
        row=row, col=1
    )

# Adding traces for each date range with unique colors and legend names
add_traces(1, dec_dates, 'navy', 'maroon', 'Actual - Nov', 'Predicted - Nov')
add_traces(2, jan_dates, 'darkgreen', 'darkorange', 'Actual - Jan', 'Predicted - Jan')
add_traces(3, jul_dates, 'purple', 'gold', 'Actual - Jul', 'Predicted - Jul')

# Update layout
fig.update_layout(height=900, width=700, title_text="Weather Data Analysis", showlegend=True)

# Show the figure
fig.show()