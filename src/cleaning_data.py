import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import glob

from src.constants import CLEANED_DATA_PATH, PREPROCESSING_PLOTS_PATH, PREPROCESSED_DATA_PATHS

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-bright')


def clean_company(company, threshold, side_to_leave='right'):
    if side_to_leave == 'right':
        company = company[company.timestamp > threshold].reset_index(drop=True)
    elif side_to_leave == 'left':
        company = company[company.timestamp < threshold].reset_index(drop=True)
    return company


def visualize_after_cleaning(df, company_name):
    # time interval for the given company
    df.timestamp = pd.to_datetime(df.timestamp)
    starting_year = df['timestamp'][0].year
    ending_year = df['timestamp'][len(df) - 1].year
    # plot after cleaning
    fig = plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(df['timestamp'], df['avg_price'], linewidth=2, linestyle='solid', color='orange')
    plt.title(
        '{0} stock daily average prices {1} - {2} after cleaning'.format(company_name, starting_year, ending_year))
    plt.xlabel('Date')
    plt.ylabel('Daily average price values [\$]')
    plt.legend(['{0}'.format(company_name)])
    plt.savefig(
        PREPROCESSING_PLOTS_PATH + '/{0}_stock_daily_average_prices_{1}_{2}_after_cleaning.png'.format(company_name,
                                                                                                       starting_year,
                                                                                                       ending_year));
    # plt.show()
    plt.close(fig)


BAD_COMPANIES = ['AAPL', 'BSY', 'CHK', 'DISCB', 'GIII', 'GRUB', 'MCBS',
                 'MNST', 'NFLX', 'NKE', 'NVDA', 'SCVL', 'TSLA']

for file_path in PREPROCESSED_DATA_PATHS:
    # extract company name
    company_name = file_path.split('../data/4_preprocessed_data\\')[1].replace(".csv", "")
    # print(company_name)
    if company_name not in BAD_COMPANIES:
        dst = f'{CLEANED_DATA_PATH}/{company_name}.csv'
        shutil.copyfile(file_path, dst)
        # print(dst)

CLEANED_DATA_PATHS = glob.glob(CLEANED_DATA_PATH + "/*.csv")
for file_path in tqdm(CLEANED_DATA_PATHS,
                      desc="Cleaning data"):  # change tqdm(CLEANED_DATA_PATHS, desc="Cleaning data") to CLEANED_DATA_PATHS if not working
    # extract company name
    company_name = file_path.split('../data/5_cleaned_data\\')[1].replace(".csv", "")
    # print(company_name)
    company = pd.read_csv(file_path, index_col=0)
    # print(company)
    if company_name == 'AZN':
        company = clean_company(company, threshold='2015-07-26', side_to_leave='right')
    elif company_name == 'SHOO':
        company = clean_company(company, threshold='2018-10-11', side_to_leave='right')
    elif company_name == 'CG':
        company = clean_company(company, threshold='2012-05-02', side_to_leave='right')
    elif company_name == 'SFM':
        company = clean_company(company, threshold='2013-07-31', side_to_leave='right')
    elif company_name == 'COLM':
        company = clean_company(company, threshold='2014-09-28', side_to_leave='right')
    elif company_name == 'HNNMY':
        company = clean_company(company, threshold='2010-06-06', side_to_leave='right')
    elif company_name == 'IHRT':
        company = clean_company(company, threshold='2019-07-17', side_to_leave='right')
    elif company_name == 'MRNA':
        company = clean_company(company, threshold='2018-12-06', side_to_leave='right')
    elif company_name == 'NEXT':
        company = clean_company(company, threshold='2017-07-24', side_to_leave='right')
    elif company_name == 'NWSA':
        company = clean_company(company, threshold='2013-06-31', side_to_leave='right')
    elif company_name == 'PAA':
        company = clean_company(company, threshold='2012-10-01', side_to_leave='right')
    elif company_name == 'PAGP':
        company = clean_company(company, threshold='2016-11-15', side_to_leave='right')
    elif company_name == 'PPC':
        company = clean_company(company, threshold='2009-12-28', side_to_leave='right')
    elif company_name == 'SBUX':
        company = clean_company(company, threshold='2015-04-08', side_to_leave='right')
    elif company_name == 'SMRT':
        company = clean_company(company, threshold='2020-08-22', side_to_leave='left')
    elif company_name == 'V':
        company = clean_company(company, threshold='2015-03-19', side_to_leave='right')
    elif company_name == 'WMG':
        company = clean_company(company, threshold='2020-06-02', side_to_leave='right')
    # save cleaned company
    company.to_csv(file_path)
    # visualize
    visualize_after_cleaning(company, company_name)
