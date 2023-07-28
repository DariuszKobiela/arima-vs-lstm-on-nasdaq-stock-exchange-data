import glob

DATA_PATH = '../data'
PLOTS_PATH = '../plots'
PREPROCESSING_PLOTS_PATH = PLOTS_PATH + '/PREPROCESSING_plots'
ARIMA_PLOTS_PATH = PLOTS_PATH + '/ARIMA_plots'
LSTM_PLOTS_PATH = PLOTS_PATH + '/LSTM_plots'
SAMPLES_PATH = '../samples'
ARIMA_SAMPLES_PATH = '../ARIMA_samples'
SAMPLES_PLOTS_PATH = PLOTS_PATH + '/SAMPLES_plots'

RAW_PARQUET_DATA_PATH = DATA_PATH + '/0_raw_parquet_data'
RAW_CSV_DATA_PATH = DATA_PATH + '/1_raw_csv_data'
SLIM_DATA_PATH = DATA_PATH + '/2_slim_data'
MERGED_DATA_PATH = DATA_PATH + '/3_merged_data'
PREPROCESSED_DATA_PATH = DATA_PATH + '/4_preprocessed_data'
CLEANED_DATA_PATH = DATA_PATH + '/5_cleaned_data'
MONTHLY_AGGREGATED_DATA_PATH = DATA_PATH + '/6_monthly_aggregated_data'

ARIMA_MODEL_RESULTS = '../models_results/ARIMA'
LSTM_MODEL_RESULTS = '../models_results/LSTM'

PREPROCESSED_DATA_PATHS = glob.glob(PREPROCESSED_DATA_PATH + "/*.csv")
CLEANED_DATA_PATHS = glob.glob(CLEANED_DATA_PATH + "/*.csv")


#defined variables
#here provide a list of companies TICKERS to download from https://www.nasdaq.com/
COMPANIES = [] 
ENDING_YEAR = '2021'
