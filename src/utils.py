import glob
import pickle
# from dotenv import dotenv_values
import numpy as np
import pandas as pd

from src.constants import SAMPLES_PATH


def pickle_array(array, file_name):
    with open(file_name, 'wb') as f: pickle.dump(array, f)


def unpickle_array(file_name):
    with open(file_name, 'rb') as f: array = pickle.load(f)
    return array


def check_if_sample_exists(sample_name):
    return len(glob.glob(f"{SAMPLES_PATH}/*{sample_name}.pickle")) > 0


def check_if_series_sample_exists(series_name):
    return len(glob.glob(f"{SAMPLES_PATH}/*{series_name}.csv")) > 0


# def load_env_variables():
#    return {
#        **dotenv_values(".env.AWS_ACCESS_KEY_ID"),  # load shared development variables
#        **dotenv_values(".env.AWS_SECRET_ACCESS_KEY"),  # load sensitive variables
#    }

def denormalize_y(array, x_max, x_min):
    denorm_array = np.multiply(array, (x_max - x_min)[:, None]) + x_min[:, None]
    return denorm_array


def denormalize_x(array, x_max, x_min):
    denorm_array = np.multiply(array, (x_max - x_min)[:, None]) + x_min[:, None]
    return denorm_array


def check_integrity(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    my_range = pd.date_range(
        start=df.timestamp.min(), end=df.timestamp.max(), freq='B')
    missing_dates = my_range.difference(df['timestamp'])
    passing = len(missing_dates) == 0
    if not passing:
        print(f"Missing Dates: {missing_dates}")
    return passing


def imputation(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    my_range = pd.date_range(start=df.timestamp.min(), end=df.timestamp.max(), freq='D')
    missing_dates = my_range.difference(df['timestamp'])
    df = df.append(pd.DataFrame(missing_dates, columns=['timestamp'])).sort_values('timestamp')
    return df
