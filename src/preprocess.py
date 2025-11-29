import pandas as pd
import numpy as np
import os

def load_and_preprocess(path, save_processed=None):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    expected_numeric = [
        'BHK','Size_in_SqFt','Price_in_Lakhs','Year_Built','Floor_No','Total_Floors',
        'Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility','Parking_Space'
    ]

    for col in expected_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df.get('Year_Built', 2020)

    median_pps = df['Price_per_SqFt'].median()

    df['Good_Investment'] = (
        (df['Price_per_SqFt'] <= median_pps) &
        (df.get('BHK', 3) >= 3) &
        (df['Availability_Status'] == 'Available')
    ).astype(int)

    df['Future_Price_5Y'] = (df['Price_in_Lakhs'] * ((1 + 0.08) ** 5)).round(2)

    if save_processed:
        df.to_csv(save_processed, index=False)

    return df


# ------------------- FIXED PATHS ------------------- #

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))       # → src/
    RAW = os.path.normpath(os.path.join(BASE, "..", "data", "india_housing_prices.csv"))
    OUT = os.path.normpath(os.path.join(BASE, "..", "data", "india_housing_prices_processed.csv"))

    print("Reading:", RAW)
    print("Saving:", OUT)

    df = load_and_preprocess(RAW, save_processed=OUT)
    print("Saved processed CSV with", len(df), "rows.")
