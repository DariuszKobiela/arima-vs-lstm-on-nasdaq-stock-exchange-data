import pandas as pd
import glob
import matplotlib.pyplot as plt
from src.constants import MERGED_DATA_PATH, PREPROCESSED_DATA_PATH, PREPROCESSING_PLOTS_PATH
from src.utils import check_integrity, imputation
from tqdm import tqdm

MONTHLY_TRANSACTIONS_THREASHOLD = 10
plt.style.use('fivethirtyeight')
plt.style.use('seaborn-bright')

MERGED_DATA_PATHS = glob.glob(MERGED_DATA_PATH + "/*.csv")
for file_path in tqdm(MERGED_DATA_PATHS, desc="Processing The Data"):
    # extract company name for further plots
    splitted = file_path.split("\\")[-1]
    company = splitted.split(".csv")[0]

    df = pd.read_csv(file_path,
                     usecols=['timestamp', 'avg_price', 'transactions_count'],
                     dtype={'avg_price': 'float64', 'transactions_count': 'int64'})
    df.timestamp = pd.to_datetime(df.timestamp)
    # time interval for the given company
    starting_year = df['timestamp'][0].year
    ending_year = df['timestamp'][len(df) - 1].year

    # plot before trimming
    fig1 = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(df['timestamp'], df['avg_price'], linewidth=2, linestyle='solid', color='blue')
    plt.title('{0} stock daily average prices (original data)  {1} - {2}'.format(company, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Daily average price values [\$]')
    plt.legend(['{0}'.format(company)])
    plt.savefig(PREPROCESSING_PLOTS_PATH + '/{0}_stock_daily_average_prices_original_data_{1}_{2}.png'.format(company,
                                                                                                              starting_year,
                                                                                                              ending_year));
    # plt.show()
    plt.close(fig1)

    # calculate starting point
    transactions_count_monthly_df = df.groupby(pd.Grouper(key='timestamp', freq='MS')).agg(
        monthly_transactions_days=('avg_price', 'count')
    )
    starting_point_timestamp = transactions_count_monthly_df[
        transactions_count_monthly_df.monthly_transactions_days >= MONTHLY_TRANSACTIONS_THREASHOLD].index.min()

    # plot monthly transactions   
    fig2 = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(transactions_count_monthly_df.reset_index()['timestamp'],
             transactions_count_monthly_df.reset_index()['monthly_transactions_days'],
             linewidth=2,
             linestyle='solid',
             color='red')
    plt.title('{0} number of monthly days with transactions {1} - {2}'.format(company, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Monthly number of days with transactions')
    plt.legend(['{0}'.format(company)])
    plt.axvline(x=starting_point_timestamp, c='k', linestyle='--')
    plt.savefig(PREPROCESSING_PLOTS_PATH + '/{0}_number_of_monthly_days_with_transactions_{1}_{2}.png'.format(company,
                                                                                                              starting_year,
                                                                                                              ending_year));
    # plt.show()
    plt.close(fig2)

    # trim the data
    df = df[df.timestamp >= starting_point_timestamp].reset_index(drop=True)

    # plot after trimming
    fig3 = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(df['timestamp'], df['avg_price'], linewidth=2, linestyle='solid', color='green')
    plt.title('{0} stock daily average prices {1} - {2} after trimming'.format(company, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Daily average price values [\$]')
    plt.legend(['{0}'.format(company)])
    plt.savefig(PREPROCESSING_PLOTS_PATH + '/{0}_stock_daily_average_prices_{1}_{2}_after_trimming.png'.format(company,
                                                                                                               starting_year,
                                                                                                               ending_year));
    # plt.show()
    plt.close(fig3)

    # imputation
    df = imputation(df).reset_index(drop=True)

    # interpolate
    df = df.drop(columns=['transactions_count']).interpolate(method='linear', limit_direction='both')
    assert check_integrity(df), f"Integrity check failed {company}"

    # plot after interpolation
    fig4 = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(df['timestamp'], df['avg_price'], linewidth=2, linestyle='solid', color='violet')
    plt.title(
        '{0} stock daily average prices {1} - {2} after interpolation'.format(company, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Daily average price values [\$]')
    plt.legend(['{0}'.format(company)])
    plt.savefig(
        PREPROCESSING_PLOTS_PATH + '/{0}_stock_daily_average_prices_{1}_{2}_after_interpolation.png'.format(company,
                                                                                                            starting_year,
                                                                                                            ending_year));
    # plt.show()
    plt.close(fig4)

    # save new file
    file_name = file_path.split('\\')[-1]
    new_file_path = f"{PREPROCESSED_DATA_PATH}\\{file_name}"
    df.to_csv(new_file_path)
