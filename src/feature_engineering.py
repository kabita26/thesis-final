# src/feature_engineering.py

def create_features(df):
    # Example: create lag feature and interaction term
    df['yield_per_temp'] = df['hg/ha_yield'] / df['avg_temp']
    df['rainfall_x_pesticide'] = df['average_rain_fall_mm_per_year'] * df['pesticides_tonnes']
    return df
