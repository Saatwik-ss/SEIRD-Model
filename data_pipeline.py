"""
data_pipeline.py
----------------
Acquires and preprocesses the 2014 West African Ebola Outbreak surveillance
dataset for the Republic of Sierra Leone (N = 7,092,113).

Source: Caitlin Rivers' machine-readable GitHub repository (derived from WHO situation reports)
URL: https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'country_timeseries_raw.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'outbreak_data.csv')

PRIMARY_URL = 'https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv'

POPULATION_SIERRA_LEONE = 7_092_113
START_DATE = '2014-05-25'
END_DATE = '2014-09-21' # 120 days total (inclusive)

def download_raw_data(dest_path=RAW_DATA_PATH):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[data_pipeline] Raw data already exists at: {dest_path}")
        return dest_path
    
    print(f"[data_pipeline] Downloading raw Ebola data from {PRIMARY_URL}...")
    try:
        urllib.request.urlretrieve(PRIMARY_URL, dest_path)
        print(f"[data_pipeline] Successfully saved to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"[data_pipeline] Download failed: {e}")
        raise

def process_sierra_leone_data(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    if not os.path.exists(raw_path):
        download_raw_data(raw_path)
        
    df = pd.read_csv(raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter columns for Sierra Leone
    sl = df[['Date', 'Cases_SierraLeone', 'Deaths_SierraLeone']].dropna(subset=['Cases_SierraLeone', 'Deaths_SierraLeone'])
    sl = sl.sort_values('Date').reset_index(drop=True)
    
    # Define exact 120-day outbreak window: May 25, 2014 to September 21, 2014
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    
    # Build a continuous daily index
    daily_index = pd.date_range(start_ts, end_ts, freq='D')
    daily_df = pd.DataFrame({'Date': daily_index})
    
    merged = pd.merge(daily_df, sl, on='Date', how='left')
    
    # Linearly interpolate cumulative cases and deaths between reporting dates
    merged['cumulative_cases'] = merged['Cases_SierraLeone'].interpolate(method='linear').bfill().ffill().round().astype(int)
    merged['cumulative_deaths'] = merged['Deaths_SierraLeone'].interpolate(method='linear').bfill().ffill().round().astype(int)
    
    # Ensure monotonically non-decreasing cumulative series
    merged['cumulative_cases'] = np.maximum.accumulate(merged['cumulative_cases'])
    merged['cumulative_deaths'] = np.maximum.accumulate(merged['cumulative_deaths'])
    
    merged['day'] = range(len(merged))
    merged['date'] = merged['Date'].dt.strftime('%Y-%m-%d')
    
    # Daily increments
    merged['daily_cases'] = merged['cumulative_cases'].diff().fillna(merged['cumulative_cases'].iloc[0]).clip(lower=0).astype(int)
    merged['daily_deaths'] = merged['cumulative_deaths'].diff().fillna(merged['cumulative_deaths'].iloc[0]).clip(lower=0).astype(int)
    
    # 7-day centered rolling average
    merged['daily_cases_7day_avg'] = merged['daily_cases'].rolling(7, center=True, min_periods=1).mean().round(1)
    
    # 75/25 chronological train/test split: Days 0-89 train (90 days), Days 90-119 test (30 days)
    split_cutoff = int(len(merged) * 0.75) # 90
    merged['split'] = ['train' if i < split_cutoff else 'test' for i in range(len(merged))]
    
    cols = [
        'day', 'date', 'daily_cases', 'daily_cases_7day_avg', 'cumulative_cases',
        'daily_deaths', 'cumulative_deaths', 'split'
    ]
    final_df = merged[cols]
    
    final_df.to_csv(output_path, index=False)
    print(f"[data_pipeline] Processed dataset saved to: {output_path}")
    print(f"[data_pipeline] Total days: {len(final_df)} | Train: {(final_df['split']=='train').sum()} | Test: {(final_df['split']=='test').sum()}")
    print(f"[data_pipeline] Cumulative Cases: {final_df['cumulative_cases'].iloc[0]} -> {final_df['cumulative_cases'].iloc[-1]}")
    print(f"[data_pipeline] Cumulative Deaths: {final_df['cumulative_deaths'].iloc[0]} -> {final_df['cumulative_deaths'].iloc[-1]}")
    return final_df

if __name__ == '__main__':
    download_raw_data()
    process_sierra_leone_data()
