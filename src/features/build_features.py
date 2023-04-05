#data packages
import pandas as pd

#packages for paths
import os
from pathlib import Path

#Logs
import logging
logging.basicConfig(level=logging.INFO)

def add_delay_status_column(raw_data: pd.DataFrame) -> pd.DataFrame:
    time_difference = raw_data['Fecha-O'] - raw_data['Fecha-I']
    minutes_difference = time_difference.dt.seconds.div(60).astype(int)
    raw_data['delay_status'] = minutes_difference > 15
    raw_data.loc[raw_data['Fecha-O'] < raw_data['Fecha-I'], 'delay_status'] = False
    return raw_data

def add_season_status_column(raw_data: pd.DataFrame) -> pd.DataFrame:
    # High Season cases:
    # Dec 15 to Dec 31
    # Jan 1 to Mar 3
    # Jul 15 to Jul 31
    # Sep 11 to Sep 30 
    high_season_ranges = [
        ((12, 15), (12, 31)),
        ((1, 1), (3, 3)),
        ((7, 15), (7, 31)),
        ((9, 11), (9, 30))
    ]
    
    raw_data['is_high_season'] = raw_data['Fecha-I'].apply(
        lambda date: any(
            start <= (date.month, date.day) <= end
            for start, end in high_season_ranges
        )
    )
    
    return raw_data

def add_day_period_column(raw_data: pd.DataFrame) -> pd.DataFrame:
    # ranges by hours
    day_periods = {
        'morning': (5, 12),
        'afternoon': (12, 19),
        'evening_one': (19, 24),
        'evening_two': (0, 5)
    }
    
    raw_data['day_period'] = raw_data['Fecha-I'].apply(
        lambda datetime: next((period for period, hours in day_periods.items() if hours[0] <= datetime.hour < hours[1]), None)
    )

    raw_data.loc[raw_data['day_period'].isin(['evening_one', 'evening_two']), 'day_period'] = 'evening'
    
    return raw_data

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    preprocessed_data = (
        raw_data
        .drop(columns=['Ori-I', 'Ori-O', 'Vlo-I', 'Vlo-O', 'OPERA', 'SIGLAORI', 'SIGLADES'], errors='ignore')
        .dropna()
        .astype({
            'Fecha-I': "datetime64",
            'Fecha-O': "datetime64",
        })
    )
    return preprocessed_data

def process_data(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    processed_data = (
        preprocessed_data
        .pipe(add_day_period_column)
        .pipe(add_season_status_column)
        .pipe(add_delay_status_column)
        .drop(columns=['Fecha-I', 'Fecha-O'], errors='ignore')
    )
    logging.info(f"processed data shape: {processed_data.shape}")
    return processed_data

if __name__ == "__main__":
    # get path to raw_data
    main_path = Path(__file__).parent.parent.parent
    raw_data_path = main_path / "data/raw/dataset_SCL.csv"

    # extract raw_data
    raw_data = pd.read_csv(raw_data_path, low_memory=False)

    # process raw_data
    preprocessed_data = preprocess_data(raw_data=raw_data)
    processed_data = process_data(preprocessed_data=preprocessed_data)

    # save processed data
    processed_data_path = main_path / "data/processed/features.csv"
    processed_data.to_csv(processed_data_path, index=False)



