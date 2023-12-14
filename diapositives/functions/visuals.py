import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt


def visualize_df(df, city, nom_poll):
    """visualize the dataframe"""

    # Filter and sort DataFrame by timestamp
    viz_df = df[(df["city"] == city) & (df["nom_poll"] == nom_poll)].sort_values(
        by="start"
    )
    # Handle double values at same date
    viz_df = viz_df.groupby('start')['values'].mean().reset_index()

    # Extract the timestamp and value columns
    timestamps = viz_df["start"]
    values = viz_df['values']

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, values, marker="o", linestyle="-")
    plt.title(f"Time Series {city}")
    plt.xlabel("Timestamp")
    plt.ylabel(f"{nom_poll} Value")
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()
