# src/data_preprocessing.py

import pandas as pd
import os

def load_and_filter_data(input_path, output_path):
    required_columns = [
        'Area', 'Item', 'Year', 'hg/ha_yield',
        'avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes'
    ]

    df = pd.read_csv(input_path)

    # Filter only rice-related rows
    rice_df = df[df['Item'].str.lower() == 'rice']

    # Keep only required columns
    rice_df = rice_df[required_columns]

    # Drop missing values
    rice_df = rice_df.dropna()

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rice_df.to_csv(output_path, index=False)
    return rice_df
