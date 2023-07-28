import pandas as pd
import glob
import matplotlib.pyplot as plt

from constants import MONTHLY_AGGREGATED_DATA_PATH, PREPROCESSED_DATA_PATHS, PREPROCESSING_PLOTS_PATH

for file_path in PREPROCESSED_DATA_PATHS:
    df = pd.read_csv(file_path, index_col=0)
    df.timestamp = pd.to_datetime(df.timestamp)
    df.sort_values(by='timestamp', axis=0, inplace=True)
    df = df.groupby(pd.Grouper(key='timestamp', freq='1M')).agg(
                avg_price = ('avg_price', 'mean')
            )
    #df.index = df.index.strftime('%Y-%m') #optional change of formatting
    new_file_path = MONTHLY_AGGREGATED_DATA_PATH + '\\' + file_path.split('\\')[-1]
    df.to_csv(new_file_path)
    #extract company name for further plots
    splitted = file_path.split("\\")[-1]
    company = splitted.split(".csv")[0]
    #time interval for the given company
    starting_year = df.index[0].year
    ending_year = df.index[len(df)-1].year
    #visualize
    plt.style.use('fivethirtyeight')
    plt.style.use('seaborn-bright')
    fig = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(df.index, df['avg_price'], linewidth=2, linestyle='solid', color='black')
    plt.title('{0} stock monthly average prices {1} - {2}'.format(company, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Monthly average price values')
    plt.legend(['{0}'.format(company)])
    plt.savefig(PREPROCESSING_PLOTS_PATH + '/{0}_stock_monthly_average_prices_{1}_{2}.png'.format(company, starting_year, ending_year))
    #plt.show()
    plt.close(fig)