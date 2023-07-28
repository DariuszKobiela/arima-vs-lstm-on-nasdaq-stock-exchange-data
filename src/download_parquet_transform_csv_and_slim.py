import pandas as pd
import glob
import boto3
import os
import shutil
import re
from datetime import datetime
import matplotlib.pyplot as plt

from constants import RAW_PARQUET_DATA_PATH, RAW_CSV_DATA_PATH, SLIM_DATA_PATH
from constants import COMPANIES, ENDING_YEAR

from passwords import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

COMPANIES_LIST = [] #should remain empty

# DEFINE DATA LOCATION IN AWS BUCKET
s3 = boto3.resource('s3',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY)

bucket = s3.Bucket('aws-bucket')

# PREPARE COMPANIES TO DOWNLOAD
for COMPANY in COMPANIES:
    if os.path.exists(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY)):
        shutil.rmtree(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY))
    os.mkdir(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY))
        
    print('Downloading trades for company {0}...'.format(COMPANY))

    for obj in bucket.objects.filter(Prefix='trades/symbol={0}'.format(COMPANY)): 
        path, filename = os.path.split(obj.key)
        bucket.download_file(obj.key, RAW_PARQUET_DATA_PATH + '/{0}/'.format(COMPANY) + filename)
        print('Trades: {0}:{1}'.format(bucket.name, obj.key))
        # check the year
        files = os.listdir(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY)) 
        number_files = len(files)
        print(number_files)
        if (number_files>6):
            break

    m = re.search('(date=)(\d{8})', obj.key)
    STARTING_YEAR = m.group(2)[:4]
    COMPANIES_LIST.append((COMPANY, STARTING_YEAR, ENDING_YEAR))
    print('\n')
    print('---------------------------------------------------------------------------------------------')
    print('Test for {0} has ended!'.format(COMPANY))
    print('---------------------------------------------------------------------------------------------')
    print('\n')
    
 # DOWNLOAD_PARQUET + PAR-> CSV + CSV SLIM
for COMPANY in COMPANIES_LIST:    
    STARTING_YEAR = COMPANY[1]
    ENDING_YEAR = COMPANY[2]
    YEARS = []
    YEAR = int(STARTING_YEAR)
    END_YEAR = int(ENDING_YEAR)
    while (YEAR <= END_YEAR):
        YEARS.append('{0}'.format(YEAR))
        YEAR+=1    
    
    if os.path.exists(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY[0])):
            shutil.rmtree(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY[0]))
    os.mkdir(RAW_PARQUET_DATA_PATH + '/{0}'.format(COMPANY[0]))
    
    #0. MEASURE EXECUTION TIME
    start_time = datetime.now()
    
    #1. DOWNLOAD PARQUET
    for year in YEARS: 
        os.mkdir(RAW_PARQUET_DATA_PATH + '/{0}/{0}{1}'.format(COMPANY[0], year))
        print('Downloading trades for company {0}, year {1}...'.format(COMPANY[0], year))

        for obj in bucket.objects.filter(Prefix='trades/symbol={0}/date={1}'.format(COMPANY[0], year)): 
            path, filename = os.path.split(obj.key)
            bucket.download_file(obj.key, RAW_PARQUET_DATA_PATH + '/{0}/{0}{1}/'.format(COMPANY[0], year) + filename)
            print('Trades: {0}:{1}'.format(bucket.name, obj.key))

        print('\n')
        print('---------------------------------------------------------------------------------------------')
        print('All trades for company {0}, year {1} downloaded!!'.format(COMPANY[0], year))
        print('---------------------------------------------------------------------------------------------')
        print('\n')
    #1.2. Save info to log file
    #now = datetime.now()
    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #with open("../log_file.txt", "a") as file_object:
    #    file_object.write('Companies downloaded in parquet format at {0}: \n'.format(dt_string))
    #    file_object.write('{0}\n'.format(COMPANIES_LIST))
    
    #2. TRANSFORM FROM PARQUET TO CSV
    for year in YEARS:
        first = True
        counter = 0
        global df
        for file in os.listdir(RAW_PARQUET_DATA_PATH + '/{0}/{0}{1}'.format(COMPANY[0], year)):     
            if file.endswith('.parquet'):
                filename = file
                source = pd.read_parquet(RAW_PARQUET_DATA_PATH + '/{0}/{0}{1}/'.format(COMPANY[0], year) + filename)
                if first:
                    df = source
                    df.to_csv(RAW_CSV_DATA_PATH + '/{0}{1}.csv'.format(COMPANY[0], year))
                    print('First parquet file from year {0} converted to csv!'.format(year))
                    counter += 1
                    first = False
                else:
                    df = source
                    df.to_csv(RAW_CSV_DATA_PATH + '/{0}{1}.csv'.format(COMPANY[0], year), mode='a', header=False)
                    counter += 1 
                    if ((counter % 100) == 0):
                        print('-------------------------------------------------')
                        print('Counter: ' + str(counter))
                        print('Next 100 parquet files appended to csv file!')  
                        print('-------------------------------------------------')
    #2.2. Save info to log file
    #now = datetime.now()
    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #with open("../log_file.txt", "a") as file_object:
    #    # Append at the end of file
    #    file_object.write('Companies transformed to csv format at {0}: \n'.format(dt_string))
    #    file_object.write('{0}\n'.format(COMPANIES_LIST))
    
    #3. TRANSFORM TO CSV SLIM
    RAW_CSV_DATA_FILES_PATHS = glob.glob(RAW_CSV_DATA_PATH + "/*.csv")
    for file_path in RAW_CSV_DATA_FILES_PATHS:
        df = pd.read_csv(file_path, usecols=['timestamp', 'price', 'canceled'], dtype={'timestamp': 'object', 'price': 'float64', 'canceled': 'bool'})
        df.timestamp = pd.to_datetime(df.timestamp)
        df.sort_values(by='timestamp', axis=0, inplace=True)
        #filtrowanie transakcji, ktore zostaly anulowane
        df = df[df.canceled == False]
        df.drop(columns=['canceled'], axis=1, inplace=True)
        #count number of daily transactions
        df = df.groupby(pd.Grouper(key='timestamp', freq='D')).agg(
            avg_price = ('price', 'mean'),
            transactions_count = ('price', 'count')
        )
        new_file_path = SLIM_DATA_PATH + '\\' + file_path.split('\\')[-1]
        df.to_csv(new_file_path)
    
    #3.2. REMOVE RAW_PARQUET_DATA_PATH content
    files = glob.glob(RAW_PARQUET_DATA_PATH + '/*')
    for f in files:
        shutil.rmtree(f)
        
    #3.3. REMOVE RAW_CSV_DATA_PATH content
    files = glob.glob(RAW_CSV_DATA_PATH + '/*')
    for f in files:
        os.remove(f)
        
    #4. MEASURE SCRIPT EXECUTION TIME
    end_time = datetime.now()
    execution_time = end_time - start_time
    with open("../log_file.txt", "a") as file_object:
        file_object.write('{0} execution time duration: {1} h:min:s.milis\n'.format(COMPANY[0], execution_time))