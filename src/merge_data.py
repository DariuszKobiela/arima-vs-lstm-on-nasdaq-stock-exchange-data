import pandas as pd
import glob

from constants import SLIM_DATA_PATH, MERGED_DATA_PATH

SLIM_DATA_FILES_PATHS = glob.glob(SLIM_DATA_PATH + "/*.csv")
SLIM_DATA_PATHS_DF = pd.DataFrame(SLIM_DATA_FILES_PATHS, columns=['path'])
SLIM_DATA_PATHS_DF['file_name'] = SLIM_DATA_PATHS_DF.path.str.split('\\', expand=True).iloc[:, -1]
SLIM_DATA_PATHS_DF[['company', 'year']] = SLIM_DATA_PATHS_DF.file_name.str.split('(\w+)(\d{4}).csv', expand=True).iloc[:,1:-1]

for company in SLIM_DATA_PATHS_DF.company.unique():
    df = pd.DataFrame()
    for file_path in SLIM_DATA_PATHS_DF[SLIM_DATA_PATHS_DF.company==company].path:
        partial_df = pd.read_csv(file_path, usecols=['timestamp', 'avg_price', 'transactions_count'], dtype={'avg_price': 'float64'})
        partial_df.timestamp = pd.to_datetime(partial_df.timestamp)
        df = df.append(partial_df)
    df.sort_values(by='timestamp', axis=0, inplace=True)
    new_file_path = f"{MERGED_DATA_PATH}\\{company}.csv" 
    df.to_csv(new_file_path)        
