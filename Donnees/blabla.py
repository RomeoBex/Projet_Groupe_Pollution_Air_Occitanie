import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
from plotly.subplots import make_subplots
#%%
weather_file_path = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(weather_file_path, delimiter=',')
df.head()
#%%
# Describe the 'nom_dept' column
nom_dept_description = df['nom_dept'].describe()
nom_dept_description
#%%
# Filter the DataFrame for rows where 'nom_dept' is 'HERAULT'
df_HAUTE_GARONNE = df[df['nom_dept'] == 'HAUTE-GARONNE']

# Display the first few rows of the new DataFrame
df_HAUTE_GARONNE.head()
# %%
df_HAUTE_GARONNE()

# %%
# Save df_HERAULT as a CSV file
output_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/df_HAUTE_GARONNE.csv'
df_HAUTE_GARONNE.to_csv(output_file_path, index=False)
# %%
# Load the weather data
weather_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Haute_Garonne.csv'
weather_data = pd.read_csv(weather_file_path, delimiter=';')

# Convert 'date_debut' in df_HERAULT and 'Date' in weather_data to datetime
df_HAUTE_GARONNE['date_debut'] = pd.to_datetime(df_HAUTE_GARONNE['date_debut'], utc=True)
weather_data['Date'] = pd.to_datetime(weather_data['Date'], utc=True)

# Convert both indices to timezone-naive datetime (remove timezone)
df_HAUTE_GARONNE['date_debut'] = df_HAUTE_GARONNE['date_debut'].dt.tz_localize(None)
weather_data['Date'] = weather_data['Date'].dt.tz_localize(None)

# Set 'date_debut' and 'Date' as the indices
df_HAUTE_GARONNE.set_index('date_debut', inplace=True)
weather_data.set_index('Date', inplace=True)

# Merge the datasets
combined_data = pd.merge(df_HAUTE_GARONNE, weather_data, left_index=True, right_index=True, how='inner')

# Check the combined data
print(combined_data.head())

# %%
# Check for missing values
missing_values = combined_data.isnull().sum()
missing_values
#%%
# Explore data statistics
combined_stats = combined_data.describe()
combined_stats
# %%
# Print column names of the combined_data DataFrame
print(combined_data.head())
#%%
# Assuming df_HERAULT and weather_data are your original datasets

# Reset index of df_HERAULT if 'date_debut' is set as index
if isinstance(df_HAUTE_GARONNE.index, pd.DatetimeIndex):
    df_HAUTE_GARONNE = df_HAUTE_GARONNE.reset_index()

# Now merge df_HERAULT with weather_data
# Assuming the weather data is already aggregated to the same time frequency (e.g., monthly)
combined_data = pd.merge(df_HAUTE_GARONNE, weather_data, left_on='date_debut', right_index=True, how='inner')

# Check if 'date_debut' is now in the combined_data
print(combined_data.columns)

# Proceed with selecting your relevant columns
relevant_columns = ['date_debut', 'nom_poll', 'valeur', 'date_fin', 'Température', 'Humidité', 
                    'Vitesse du vent moyen 10 mn', 'nom_station', 'typologie', 'influence']
sub_dataframe = combined_data[relevant_columns]

# Display the first few rows of the sub-dataframe
sub_dataframe.head()

#%%

# %%
# Check for missing values
missing_values = sub_dataframe.isnull().sum()
print("Missing Values:\n", missing_values)

#%%
# Identify numeric columns
numeric_cols = sub_dataframe.select_dtypes(include=[np.number]).columns

# Fill missing values in numeric columns with the mean of their respective column
sub_dataframe[numeric_cols] = sub_dataframe[numeric_cols].fillna(sub_dataframe[numeric_cols].mean())

# Check for missing values again
missing_values_after = sub_dataframe.isnull().sum()
print("Missing Values after filling with mean:\n", missing_values_after)
# %%
# Basic descriptive statistics
descriptive_stats = sub_dataframe.describe()
print(descriptive_stats)
#%%