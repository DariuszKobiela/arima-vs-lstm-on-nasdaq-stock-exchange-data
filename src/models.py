import math

from pandas import DateOffset
from tqdm import tqdm

# set global random_seed
import tensorflow as tf

from src.utils import denormalize_y, denormalize_x

tf.random.set_seed(7)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.python.keras.callbacks import EarlyStopping
import datetime as dt

# ARIMA imports
import pandas as pd
import glob
import numpy as np
import glob
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA  # here the arima.model instead of arima_model

import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
# warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)
import statsmodels.api as sm
from datetime import datetime
import os

from src.constants import SAMPLES_PATH, ARIMA_SAMPLES_PATH, ARIMA_PLOTS_PATH, ARIMA_MODEL_RESULTS, LSTM_PLOTS_PATH, \
    LSTM_MODEL_RESULTS


# from statsmodels.tsa.arima_model import ARIMA #usually I used this version, but it will be deprecated
# C:\Users\Darek_PC\miniconda3\lib\site-packages\statsmodels\tsa\arima_model.py:472: FutureWarning:
# statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
# been deprecated in favor of statsmodels.tsa.arima.model.ARIMA (note the .
# between arima and model) and
# statsmodels.tsa.SARIMAX. These will be removed after the 0.12 release.

# statsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and
# is both well tested and maintained.

# To silence this warning and continue using ARMA and ARIMA until they are
# removed, use:

# import warnings
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
# warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)

class Model:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name


class ARIMAModel(Model):
    def __init__(self, experiment_name, experiment_id, output_window_size, series_averaging=False, p_range=range(0, 5), q_range=range(0, 5),
                 PERIOD=365,
                 P_VALUE_PERCENT='5%', REGRESSION_TYPE_FOR_DF_TEST='c', AUTOLAG_FOR_DF_TEST='AIC',
                 LJUNGBOX_BOXPIERCE_LAGS=24):
        super().__init__(experiment_name)
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.series_averaging = series_averaging
        """
        https://predictivehacks.com/time-series-decomposition/
        Our data here are aggregated by days. 
        The period we want to analyze is by year so that’s why we set the period to 365.
        """
        self.period = PERIOD
        self.p_range = p_range
        self.q_range = q_range
        self.p_value_percent = P_VALUE_PERCENT  # ['1%', '5%', '10%']
        self.p_value = float(P_VALUE_PERCENT.strip('%')) / 100.0
        """
        'c' : constant only, the most common choice (default value)
        'ct' : constant and trend.
        'ctt' : constant, and linear and quadratic trend
        'n' : no constant, no trend
        """
        self.regression_type_for_df_test = REGRESSION_TYPE_FOR_DF_TEST  # ['c', 'ct', 'ctt', 'n']
        self.autolag_for_df_test = AUTOLAG_FOR_DF_TEST  # ['AIC', 'BIC']
        self.ljungbox_boxpierce_lags = LJUNGBOX_BOXPIERCE_LAGS  # for NOT seasonal series number of lags is always 24
        self.periods_to_forecast = output_window_size  # a list like [6, 10, 30]

    def ensure_stationarity_and_perform_forecast(self):
        COLUMN_NAMES = ['company', 'experiment_id', 'model_name', 'p', 'd', 'q', 'fc_period', 'MSE', 'MAPE',
                        'exec_time_hours', 'exec_time']
        mse_df = pd.DataFrame(columns=COLUMN_NAMES)
        if self.series_averaging == True:
            paths = glob.glob(ARIMA_SAMPLES_PATH + "/*series_averaging_True_use_log_True.csv")
        else:
            paths = glob.glob(ARIMA_SAMPLES_PATH + "/*series_averaging_False_use_log_True.csv")
        # tmp_counter = 0
        for path in tqdm(paths, desc="Predicting stock prices using ARIMA"):
            # SET PLOTS STYLE
            plt.style.use('fivethirtyeight')
            plt.style.use('seaborn-bright')
            # extract company name for further plots
            splitted = path.split("\\")[-1]
            company = splitted.split(".csv")[0]
            company_short = company.split("_")[0]
            print("company", company)
            # print("company_short", company_short)
            df = pd.read_csv(path, index_col=0)
            df.timestamp = pd.to_datetime(df.timestamp)
            df.sort_values(by='timestamp', inplace=True)
            # print(df)
            # NO LONGER NEEDED: calculate weight for the given company
            # weight = df.avg_price.max() - df.avg_price.min()
            # print(f'{company_short} weight: {weight} \n')
            # 1. Time Series decomposition
            try:
                self.decompose_time_series(df.copy(), company, company_short, self.period)
            except ValueError:
                print(f"Exception: Decomposition not possible - too short series {company}")
            except:
                print(f"Some error occurred for company {company}")
            # 3. Stationarity
            test_statistic, critical_value = self.perform_df_test(df.avg_price, self.regression_type_for_df_test,
                                                                  self.autolag_for_df_test, self.p_value_percent)
            # print(test_statistic, critical_value)
            number_of_first_diff = 0
            if test_statistic > critical_value:
                while (test_statistic > critical_value):
                    number_of_first_diff += 1
                    df_diff = self.perform_first_difference(df.copy(), number_of_first_diff, company, company_short)
                    test_statistic, critical_value = self.perform_df_test(df_diff.diff_price,
                                                                          self.regression_type_for_df_test,
                                                                          self.autolag_for_df_test,
                                                                          self.p_value_percent)
                    #print(f"Number of first diff = {number_of_first_diff}")
                    #print(test_statistic, critical_value)
                    if number_of_first_diff > 2:
                        print(f"WARNING! SOMETHING IS WRONG - THERE WAS {number_of_first_diff} first difference")
            else:
                # if no first diff performed - we still need df_diff for further computations
                df_diff = df.copy()
                df_diff = df_diff.rename(columns={'avg_price': 'diff_price'})
            print(f"Number of performed first diff = {number_of_first_diff}")
            # print('Original dataframe', df)
            # print('First differenced dataframe', df_diff)
            # EXTRA STEP: SAVE ACF and PACF chart
            """
            They are used to choose p, q paramaters: usually we count up to 10th tab, how many of them is above the significance level. 
            They are also used to check stationarity: if the tabs come down quickly, then this suggests that the series should be stationary.
            """
            try:
                self.create_acf_pacf_charts(df_diff.diff_price, self.ljungbox_boxpierce_lags, 'timeseries', company,
                                            company_short, use_pq=False)
            except:
                print(f"Exception: not possible to create acf_pacf_charts for {company}")
            # 4. Automatically choose ARIMA parameters p, d, q (measure "training" time)
            """
            IMPORTANT!! I PERFORM ARIMA ON DIFFERENCED AND LOGGED VALUES!!
            Only log values was not enough (not stationary).
            """
            start_time = datetime.now()
            d = number_of_first_diff
            p_range = self.p_range  # range(0, 10)
            q_range = self.q_range  # range(0, 10)
            # POSSIBLE ARIMA COMBINATIONS - ANALIZA AUTOMATYCZNA
            arima_combinations_list = []
            aic_results_list = []
            bic_results_list = []
            p_parameter_list = []
            d_parameter_list = []
            q_parameter_list = []
            # hqic_results_list = []
            for p in p_range:
                for q in q_range:
                    results = ARIMA(df_diff.diff_price, order=(p, d, q),
                                    enforce_stationarity=False).fit()  # method='innovations_mle'; disp=-1
                    # df_diff.diff_price, initialization='approximate_diffuse'
                    # , enforce_stationarity=False should help
                    # https://github.com/statsmodels/statsmodels/issues/5459
                    arima_combinations_list.append(f"ARIMA({p},{d},{q})")
                    p_parameter_list.append(p)
                    d_parameter_list.append(d)
                    q_parameter_list.append(q)
                    aic_results_list.append(results.aic)
                    bic_results_list.append(results.bic)
                    # hqic_results_list.append(results.hqic)
            #print(aic_results_list)
            #print(bic_results_list)
            # print(hqic_results_list)
            auto_ARIMA_df = pd.DataFrame(list(zip(arima_combinations_list,
                                                  p_parameter_list,
                                                  d_parameter_list,
                                                  q_parameter_list,
                                                  aic_results_list,
                                                  bic_results_list)),
                                         columns=['ARIMA', 'p', 'd', 'q', 'AIC', 'BIC'])
            #print("auto_ARIMA_df", auto_ARIMA_df)
            # Save model results to CSV
            auto_ARIMA_df.to_csv(f'{ARIMA_MODEL_RESULTS}/{self.experiment_name}_{company}_ARIMA_parameters.csv')
            #auto_ARIMA_df.to_csv(ARIMA_MODEL_RESULTS + '/{0}_ARIMA_parameters.csv'.format(company))

            #print('Best ARIMA according to AIC criterium:')
            best_ARIMA_aic = auto_ARIMA_df.sort_values(by=['AIC']).iloc[0]
            print(best_ARIMA_aic)
            p_best_aic = best_ARIMA_aic[1]
            q_best_aic = best_ARIMA_aic[3]

            #print('\nBest ARIMA according to BIC criterium:')
            best_ARIMA_bic = auto_ARIMA_df.sort_values(by=['BIC']).iloc[0]
            print(best_ARIMA_bic)
            p_best_bic = best_ARIMA_bic[1]
            q_best_bic = best_ARIMA_bic[3]

            best_pq_list = []
            best_pq_list.append((p_best_aic, q_best_aic))
            if ((p_best_aic, q_best_aic) != (p_best_bic, q_best_bic)):
                best_pq_list.append((p_best_bic, q_best_bic))
            # measure "training" time
            end_time = datetime.now()
            execution_time = end_time - start_time
            exec_hours = execution_time.seconds // 3600
            exec_minutes = (execution_time.seconds % 3600) // 60
            exec_seconds = (execution_time.seconds % 3600) % 60
            execution_time_h = f"{exec_hours}h {exec_minutes}min {exec_seconds}s"
            print(f"\nDuration of choosing the best ARIMA model: {execution_time_h}\n")
            # 5. Check the autocorrelations for residuals (white noise)
            for pq in best_pq_list:
                try:
                    lb_p_value, bp_p_value, residuals = self.perform_ljungbox_boxpierce(df.avg_price, int(pq[0]), d,
                                                                                        int(pq[1]),
                                                                                        self.ljungbox_boxpierce_lags)
                    self.create_acf_pacf_charts(residuals, self.ljungbox_boxpierce_lags, 'residuals', company,
                                                company_short, use_pq=True, pq=pq)
                    if (self.p_value > lb_p_value) or (self.p_value > bp_p_value):
                        with open(
                                f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_residuals_p_{pq[0]}_q_{pq[1]}_plot_ACF_and_PACF.txt',
                                'a') as f:
                            f.write(
                                f'{self.experiment_name},{company_short},residuals p={pq[0]},q={pq[1]},chosen p_value level={self.p_value},llungbox p_value={round(lb_p_value, 4)}, boxpierce p_value={round(bp_p_value, 4)}\n')
                            f.close()
                        # img = mpimg.imread(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_residuals_p_{pq[0]}_q_{pq[1]}_plot_ACF_and_PACF.png', format='png')
                        # #img = mpimg.imread(ARIMA_PLOTS_PATH + '/{0}_{1}_plot_ACF_and_PACF.png'.format(company, 'residuals'), format='png')
                        # # Output Images
                        # fig = plt.figure(figsize=(18, 12), dpi=140)
                        # plt.axis('off')  # turn off axis
                        # plt.title(f'Please decide, if the residuals are white noise.\n \
                        #     When you are ready, close the window and choose. \n \
                        #     Llungbox p_value = {round(lb_p_value, 4)}, Boxpierce p_value = {round(bp_p_value, 4)}. \n \
                        #     Chosen p_value level is {self.p_value}')
                        # plt.imshow(img)
                        # plt.show()
                        # # USER HAS TO CHOOSE, IF THE SERIES IS STATIONARY
                        # # TODO: implement tkinter, when this will be a python script
                        # answer = input('Do the residuals approximate white noise? [y/n]: ')
                        # if answer == 'n':
                        #     print('IT IS NOT POSSIBLE TO PERFORM THE FORECAST!\n')
                        #     print('THE MODEL IS ABANDONED!\n')
                        #     # TODO: implement some info about this to log file
                        #     continue  # this will end this for loop iteration
                except ValueError:
                    print(f"An ljungbox-boxpierce exception occurred for company {company}, parameters pq {pq}")
                except:
                    print(f"Some error occurred for company {company}, parameters pq {pq}")
                # The residuals of the model are white noise. We can proceed.
                # 6. PERFORM ARIMA FORECAST
                try:
                    for period in self.periods_to_forecast:
                        # Prepare train and test set
                        # df.index = df.timestamp
                        # Używamy wartości tylko po logarytmie, żeby inwestor miał prawdziwe wartości (exp(log_value))
                        train = df.avg_price[:-period]
                        test = df.avg_price[-period:]
                        test_timestamp = df.timestamp[-period:]
                        train_timestamp = df.timestamp[:-period]

                        # define lists to be predicted
                        history = [x for x in train]
                        future = [x for x in test]
                        predictions = []
                        se_list = []
                        lower_series_list = []
                        upper_series_list = []

                        # Train the model separately for every prediction
                        # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
                        for per in range(period):
                            model = ARIMA(history, order=(pq[0], d, pq[1]))
                            fitted = model.fit()  # transparams=False
                            fcast_res = fitted.get_forecast(steps=1, alpha=self.p_value)
                            # print("Forecast results for per={0}:".format(per))
                            # print(fcast_res.summary_frame())
                            fc = fcast_res.summary_frame()['mean'][0]
                            predictions.append(fc)
                            se = fcast_res.summary_frame()['mean_se'][0]
                            se_list.append(se)
                            lower_series = fcast_res.summary_frame()['mean_ci_lower'][0]
                            lower_series_list.append(lower_series)
                            upper_series = fcast_res.summary_frame()['mean_ci_upper'][0]
                            upper_series_list.append(upper_series)
                            # append to the test set
                            obs = future[per]
                            history.append(obs)
                            # print(f"Forecast results for per={per}: predicted={fcast_res.summary_frame()['mean'].values[0]}, expected={obs}")
                            # print('predicted=%f, expected=%f' % (fcast_res.summary_frame()['mean'], obs))
                        # # print predictions
                        # prediction_df = pd.DataFrame(
                        #     {'prediction': predictions,
                        #      'se': se_list,
                        #      'lower_series': lower_series_list,
                        #      'upper_series': upper_series_list
                        #      })
                        # print(prediction_df)
                        # print model summary
                        # print("Summary of the last predicted model", fitted.summary())
                        # save summary to file
                        _path = f'{ARIMA_MODEL_RESULTS}/{self.experiment_name}_{company}_ARIMA({pq[0]}{d}{pq[1]})_model_summary_period_{period}.csv'
                        with open(_path, 'w') as f:
                            f.write(fitted.summary().as_csv())

                        # Test the model (perform forecast)
                        # https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html
                        # fcast_res = fitted.get_forecast(steps=period, alpha=P_VALUE)  # 3 okresy do przodu, 95% conf
                        # fc = fcast_res.summary_frame()['mean']
                        # se = fcast_res.summary_frame()['mean_se']
                        # lower_series = fcast_res.summary_frame()['mean_ci_lower']
                        # upper_series = fcast_res.summary_frame()['mean_ci_upper']
                        # get_forecast instead of forecast
                        # se - błąd standardowy, standard error
                        # fc - forecast
                        # conf - poziom ufności, confidence interval
                        # Make as pandas series - NOT NEEDED ANY MORE
                        # fc_series = pd.Series(fc, index=test.index)
                        # lower_series = pd.Series(conf[:, 0], index=test.index)
                        # upper_series = pd.Series(conf[:, 1], index=test.index)

                        # Make plots
                        self.create_prediction_plot(df, train, test, predictions, lower_series_list, upper_series_list,
                                                    period, train_timestamp, test_timestamp, company, company_short,
                                                    p=pq[0], d=d, q=pq[1])
                        self.create_zoomed_prediction_plot(df, test, predictions, lower_series_list, upper_series_list,
                                                           period, test_timestamp, company, company_short, p=pq[0], d=d,
                                                           q=pq[1])
                        # Calculate MSE
                        mse = np.mean(np.square(np.exp(predictions) - np.exp(test.values)))
                        # Calculate MAPE
                        mape = np.mean(np.abs((np.exp(test.values) - np.exp(predictions)) / np.exp(test.values))) * 100
                        # def mean_absolute_percentage_error(y_true, y_pred):
                        #    y_true, y_pred = np.array(y_true), np.array(y_pred)
                        #    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                        # 7. SAVE RESULTS
                        # 7.1 results of specific experiment
                        prediction_df = pd.DataFrame(
                            {'company': company_short,
                             'real_value': test.values,
                             'prediction': predictions,
                             'se': se_list,
                             'lower_series': lower_series_list,
                             'upper_series': upper_series_list
                             })
                        # save prediction results:
                        prediction_df.to_csv(
                            f'{ARIMA_MODEL_RESULTS}/{self.experiment_name}_{company}_ARIMA({pq[0]}{d}{pq[1]})_period_{period}_prediction_results.csv')
                        # forecast_results = pd.DataFrame(zip(test.values, predictions), columns=['True_Value', 'ARIMA_forecast'])
                        # forecast_results['Company'] = company_short
                        # forecast_results.to_csv(ARIMA_MODEL_RESULTS + f'/{company}_forecast_results_ARIMA({pq[0]},{d},{pq[1]})_period_{period}.csv')
                        # 7.2 MASTER TABLE
                        # [company, ARIMA, p,d,q , period, mse]
                        arima_tmp = f'ARIMA({pq[0]},{d},{pq[1]})'
                        mse_list = [company_short, self.experiment_id, arima_tmp, pq[0], d, pq[1], period, mse, mape,
                                    execution_time_h, execution_time]
                        mse_df = mse_df.append(pd.Series(mse_list, index=mse_df.columns), ignore_index=True)
                        # print(mse_df)
                        ### END OF LOOP ###
                except ValueError:
                    print(
                        f"An ARIMA prediction exception occurred for company {company}, parameters pq {pq}, period {period}")
                except:
                    print(f"Some error occurred for company {company}, parameters pq {pq}, period {period}")
            # delete tmp_counter
            # tmp_counter += 1
            # print("counter", tmp_counter)
            # if tmp_counter == 3:
            #    break
        # 7.3 After all experiments
        saving_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        mse_df.to_csv(ARIMA_MODEL_RESULTS + f'/SEMI_MASTER_TABLE_forecast_results_{saving_time}.csv')
        #print(mse_df)
        # 8. MODEL RESULTS, MASTER TABLE WITH ALL THE DATA
        MASTER_COLUMN_NAMES = ['timestamp', 'experiment_id', 'experiment_desc', 'model_name', 'p', 'd', 'q',
                               'fc_period', 'MSE', 'MAPE', 'exec_runtime_hours', 'exec_runtime']
        master_table_df = pd.DataFrame(columns=MASTER_COLUMN_NAMES)
        ###
        arima_params = []
        for index, row in mse_df.iterrows():
            tmp = (row['p'], row['d'], row['q'])
            if tmp not in arima_params:
                arima_params.append(tmp)
        print("arima_params", arima_params)
        # mse_error_list = []
        tmp_df = mse_df.copy()
        for period in self.periods_to_forecast:
            second_df = tmp_df[tmp_df['fc_period'] == period]
            for pdq in arima_params:
                third_df = second_df[
                    (second_df['p'] == pdq[0]) & (second_df['d'] == pdq[1]) & (second_df['q'] == pdq[2])]
                #print("third_df['MSE']", third_df['MSE'])
                mse_error = third_df['MSE'].mean()
                # print('mse_error', mse_error)
                # print("third_df['MAPE']", third_df['MAPE'])
                mape_error = third_df['MAPE'].mean()
                #print('mape_error', mape_error)
                #print("third_df['exec_time']", third_df['exec_time'])
                exec_time = third_df['exec_time'].sum()  # sum rather than mean() execution time
                #print('exec_time', exec_time)
                exec_hours = exec_time.seconds // 3600
                exec_minutes = (exec_time.seconds % 3600) // 60
                exec_seconds = (exec_time.seconds % 3600) % 60
                exec_time_h = f"{exec_hours}h {exec_minutes}min {exec_seconds}s"
                print(f"Duration of choosing the best ARIMA model: {exec_time_h}")
                print(f'MSE TEST SCORE for ARIMA({pdq}) for period={period}: {mse_error}')
                print(f'MAPE TEST SCORE for ARIMA({pdq}) for period={period}: {mape_error}')
                master_table_values = [datetime.now().strftime('%Y-%m-%d_%H_%M'),
                                       self.experiment_id,
                                       self.experiment_name,
                                       f'ARIMA{pdq}',
                                       pdq[0],
                                       pdq[1],
                                       pdq[2],
                                       period,
                                       mse_error,
                                       mape_error,
                                       exec_time_h,
                                       exec_time]
                master_table_df = master_table_df.append(pd.Series(master_table_values, index=master_table_df.columns),
                                                         ignore_index=True)
        # 9. SAVE MASTER TABLE WITH ALL THE DATA
        print(master_table_df)
        master_file = ARIMA_MODEL_RESULTS + f'/MASTER_TABLE.csv'
        file_exists = os.path.isfile(master_file)
        if not file_exists:
            with open(master_file, 'x') as f:
                master_table_df.to_csv(master_file)
                f.close()
        else:
            master_table_df.to_csv(master_file, mode='a', header=None)
        ### END OF FUNCTION ###

    def decompose_time_series(self, df, company, company_short, period):
        """
        sm.tsa.seasonal_decompose returns a DecomposeResult.
        This has attributes observed, trend, seasonal and resid, which are pandas series.
        You may plot each of them using the pandas plot functionality.
        """
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        result = seasonal_decompose(df['avg_price'], period=period)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), dpi=65)
        ax1.plot(result.observed, color='blue', linewidth=1)
        ax1.set_title('{0} original time series'.format(company_short))
        ax2.plot(result.trend, color='red', linewidth=1)
        ax2.set_title('Trend')
        # ax3.plot(result.seasonal, color='red', linewidth=1)
        # ax3.set_title('Seasonality')
        ax3.plot(result.resid, color='red', linewidth=1)
        ax3.set_title('Residuals')
        plt.xlabel('Date')
        plt.savefig(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_decomposition.png');
        # plt.show()
        plt.close(fig)

    def perform_df_test(self, timeseries, reg, autol, pval):
        """
        I choose AIC as autolag (more often used than BIC).
        I choose regression type (reg) as 'c' : constant only (default).
        """
        dftest = adfuller(timeseries, regression=reg, autolag=autol)
        # p_value = dftest[1]
        test_statistic = dftest[0]
        critical_value = dftest[4][pval]
        return test_statistic, critical_value

    def perform_first_difference(self, df, number_of_first_diff, company, company_short):
        if number_of_first_diff == 1:
            df['diff_price'] = df.avg_price.diff()
            df.drop('avg_price', axis=1, inplace=True)
        else:
            df.diff_price = df.diff_price.diff()
        df = df.iloc[1:, :]
        df = df.reset_index(drop=True)
        fig = plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(df['timestamp'], df['diff_price'], color='green', linewidth=1, linestyle='solid')
        plt.title("{0} after {1} first difference".format(company_short, number_of_first_diff))
        plt.xlabel('Date')
        plt.ylabel('Average price values after log and first diff')
        plt.legend(['{0}'.format(company_short)])
        plt.savefig(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_after_{number_of_first_diff}_first_difference.png');
        #plt.savefig(ARIMA_PLOTS_PATH + '/{0}_after_{1}_first_difference.png'.format(company, number_of_first_diff));
        # plt.show()
        plt.close(fig)
        return df

    def create_acf_pacf_charts(self, series, lags, datatype, company, company_short, use_pq=False, pq=(0, 0)):
        """
        Remember to ommit first tab ("wypustka"), because it's the correlation of the variable with itself.
        It can be ommited by using parameter zero=False.
        Documentation about plot_acf: https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
        zero: bool, optional
        Flag indicating whether to include the 0-lag autocorrelation. Default is True.
        """
        # SET TEMPORARY PLOT STYLE
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white'
        # ACF
        fig = plt.figure(figsize=(12, 9), dpi=90)
        ax1 = fig.add_subplot(211)
        sm.graphics.tsa.plot_acf(series, lags=self.ljungbox_boxpierce_lags, zero=False, ax=ax1)  # residuals, fig2
        plt.ylabel('Value')
        plt.title('{0} {1} ACF'.format(company_short, datatype))
        # PACF
        ax2 = fig.add_subplot(212)
        sm.graphics.tsa.plot_pacf(series, lags=self.ljungbox_boxpierce_lags, zero=False, method="ywm",
                                  ax=ax2)  # residuals, fig3
        plt.ylabel('Value')
        plt.xlabel('Number of tabs')

        if use_pq:
            plt.savefig(
                f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_{datatype}_p_{pq[0]}_q_{pq[1]}_plot_ACF_and_PACF.png');
        else:
            plt.savefig(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_{datatype}_plot_ACF_and_PACF.png');
        # plt.savefig(ARIMA_PLOTS_PATH + '/{0}_{1}_plot_ACF_and_PACF.png'.format(company, datatype));  # , bbox_inches=0
        # plt.show()
        plt.close(fig)
        # plt.close(fig2)
        # plt.close(fig3)
        # BACK TO DEFINED STYLE
        plt.style.use('fivethirtyeight')
        plt.style.use('seaborn-bright')

    def perform_ljungbox_boxpierce(self, df_price, p, d, q, lags):
        """
        Performs Ljung-Box and Box-Pierce tests. Check the residuals autocorrelations.
        https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
        Remember, that p, d, q values HAVE TO BE INTEGERS!!
        """
        res = sm.tsa.ARIMA(endog=df_price, order=(p, d, q)).fit()  # transparams=False
        lb_bp_results = sm.stats.acorr_ljungbox(res.resid, lags=[lags], return_df=True, boxpierce=True)
        lb_p_value = lb_bp_results['lb_pvalue'].values[0]
        bp_p_value = lb_bp_results['bp_pvalue'].values[0]
        residuals = res.resid
        return lb_p_value, bp_p_value, residuals

    def create_prediction_plot(self, df, train, test, fc_series, lower_series, upper_series, period, train_timestamp,
                               test_timestamp, company, company_short, p, d, q):
        # prepare xticks for plot
        # list_x_ticks = []
        # list_timestamps_x_ticks = []
        # final_year = df.timestamp[len(df)-1].year
        # for xtick in range(0, len(df) + round(len(df)/7), round(len(df)/7)):
        #    list_x_ticks.append(xtick)
        #    if xtick > len(df):
        #        list_timestamps_x_ticks.append(final_year + 2)
        #    else:
        #        list_timestamps_x_ticks.append(df.timestamp[xtick].year)

        # # prepare yticks for plot
        # list_y_ticks = []
        # # print(df.avg_price.min(), df.avg_price.max())
        # min_val = df.avg_price.min()
        # max_val = df.avg_price.max()
        # step = (max_val - min_val) / 9
        # # print("step:", step)
        # # print('\n')
        # for ytick in np.arange(min_val, max_val + step, step):
        #     #print(ytick)
        #     list_y_ticks.append(round(ytick, 2))
        # # print("list_y_ticks", list_y_ticks)
        #
        # # prepare y_exp_ticks for plot
        # list_exp_y_ticks = []
        # # print("df.avg_price.min()", df.avg_price.min(), "df.avg_price.max()", df.avg_price.max())
        # # print("exp(min_price)", np.exp(df.avg_price.min()), "exp(max_price)", np.exp(df.avg_price.max()))
        # min_val = np.exp(df.avg_price.min())
        # max_val = np.exp(df.avg_price.max())
        # step = (max_val - min_val) / 9
        # # print("step:", step)
        # # print('\n')
        # for ytick in np.arange(min_val, max_val + step, step):
        #     # print("ytick", ytick)
        #     list_exp_y_ticks.append(round(ytick, 2))
        # # print("list_exp_y_ticks", list_exp_y_ticks)

        # Plot
        # SET TEMPORARY PLOT STYLE
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white'

        fig = plt.figure(figsize=(12, 7), dpi=90)

        plt.plot(train_timestamp, train, label='training', linewidth=1, color='b')  # train.index
        plt.plot(test_timestamp, test, label='actual', linewidth=1, color='g')  # test.index
        plt.plot(test_timestamp, fc_series, label='forecast', linewidth=1, color='r')  # fc_series.index

        plt.fill_between(test_timestamp, lower_series, upper_series, color='k', alpha=.15)  # lower_series.index,

        plt.title(f'Forecast vs Actuals for {company_short} forecast period = {period}')
        plt.legend(loc='upper left', fontsize=8)
        plt.xlabel('Time [years]')
        plt.ylabel('Real value [\$]')
        # plt.xticks(list_x_ticks, list_timestamps_x_ticks, rotation = 45)
        #plt.yticks(list_y_ticks, list_exp_y_ticks) #, color="red"
        plt.savefig(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_period_{period}_ARIMA({p}{d}{q})_prediction_plot.png');
        #plt.savefig(ARIMA_PLOTS_PATH + '/{0}_ARIMA_prediction_plot.png'.format(company))
        # plt.show()
        plt.close(fig)
        #################################################

    def create_zoomed_prediction_plot(self, df, test, fc_series, lower_series, upper_series, period, test_timestamp,
                                      company, company_short, p, d, q):
        # prepare yticks
        # minimum_value = lower_series.values.min()
        # maximum_value = upper_series.values.max()
        # minimum_value = min(lower_series)
        # maximum_value = max(upper_series)
        # # print("min value", minimum_value, "max value", maximum_value)
        #
        # # prepare yticks for plot
        # list_y_ticks = []
        # # display(minimum_value, maximum_value)
        # min_val = round(minimum_value, 2)
        # max_val = round(maximum_value, 2)
        # # print("min value", min_val, "max value", max_val)
        # step = round((max_val - min_val) / 20, 2)
        # # print("step:", step)
        # # print('\n')
        # for ytick in np.arange(min_val, max_val + step, step):
        #     # print(ytick)
        #     list_y_ticks.append(round(ytick, 2))
        # # print("final y list", list_y_ticks)
        #
        # # prepare y_exp_ticks for plot
        # list_exp_y_ticks = []
        # # print("min value", minimum_value, "max value", maximum_value)
        # min_val = round(np.exp(minimum_value), 2)
        # max_val = round(np.exp(maximum_value), 2)
        # # print("min exp value", min_val, "max exp value", max_val)
        # step = round((max_val - min_val) / 20, 2)
        # # print("step:", step)
        # # print('\n')
        # for ytick in np.arange(min_val, max_val, step):
        #     # print(ytick)
        #     list_exp_y_ticks.append(round(ytick, 2))
        # # print("final y exp list", list_exp_y_ticks)

        # PLOT
        fig = plt.figure(figsize=(12, 7), dpi=90)
        ax3 = plt.subplot(111)
        plt.plot(test_timestamp, test, label='actual', linewidth=2, color='g')  # test.index
        plt.plot(test_timestamp, fc_series, label='forecast', linewidth=2, color='r')
        plt.fill_between(test_timestamp, lower_series, upper_series, color='k', alpha=.15)

        # ax3.margins(x=-0.49, y=-0.25)
        # ax3.xaxis.zoom(4)
        plt.title(f'Forecast vs Actuals for {company_short} forecast period = {period}')
        plt.legend(loc='upper left', fontsize=8)
        plt.xticks(rotation=45)
        # plt.xticks(list_x_ticks, list_timestamps_x_ticks, rotation = 45)
        # plt.yticks(list_y_ticks[:-1], list_exp_y_ticks[:-1]) #, color="red"
        #plt.yticks(list_y_ticks, list_exp_y_ticks)
        plt.ylabel('Real value [\$]')
        plt.xlabel('Date')
        # plt.xlim(2019, 2021)
        plt.tight_layout()  # in order not to cut bottom part of plot in saved one
        plt.savefig(f'{ARIMA_PLOTS_PATH}/{self.experiment_name}_{company_short}_period_{period}_ARIMA({p}{d}{q})_zoomed_prediction_plot.png');
        #plt.savefig(ARIMA_PLOTS_PATH + '/{0}_ARIMA_zoomed_prediction_plot.png'.format(company))
        # plt.show()
        plt.close(fig)


class NeuralNetworkModel(Model):
    def __init__(self, experiment_name, experiment_id, input_window_size, output_window_size, optimizer_name,
                 activation="tanh", freq=30, input_averaging=False,
                 output_averaging=False, n_features=1, optimizer=Adam(), loss=MeanSquaredError(), use_log=True):
        super().__init__(experiment_name)
        self.model = Sequential()
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.input_averaging = input_averaging
        self.output_averaging = output_averaging
        self.freq = freq
        self.use_log = use_log
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = None
        self.n_features = n_features
        self.n_steps = input_window_size if not input_averaging else math.ceil(input_window_size / freq)
        self.output_size = output_window_size if not output_averaging else math.ceil(output_window_size / freq)
        self.architecture_name = None
        self.model_details = None
        self.optimizer_name = optimizer_name
        self.activation = activation
        self.results = None
        self.epochs = None

    def generate_saving_name(self):
        return f"{self.experiment_name}_input_window_size_{self.input_window_size}_output_window_size_{self.output_window_size}_freq_{self.freq}_n_steps_{self.n_steps}_output_size_{self.output_size}_use_log_{self.use_log}_architecture_{self.architecture_name}_optimizer_{self.optimizer_name}"

    def prepare(self, layers_list, architecture_name, optimizer=Adam(), optimizer_name='Adam', activation="tanh"):
        self.architecture_name = architecture_name
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.activation = activation
        # prepare model
        self.model = Sequential()
        for idx, layer in enumerate(layers_list):
            if idx == 0:
                layer._batch_input_shape = (None, self.n_steps, self.n_features)
            # if idx == len(layers_list)-1:
            #     layer.units = self.output_size
            self.model.add(layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mape', 'mse'])

    def fit(self, X_train, Y_train, batch_size=128, validation_split=0.2, epochs=10):
        # epochs zmień jak będziesz trenował
        #epochs = 3
        self.batch_size = batch_size
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.n_features))
        callback = EarlyStopping(monitor='loss', patience=25, restore_best_weights=True, min_delta=0.001)
        self.results = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                                      validation_split=validation_split, callbacks=[callback])
        # self.model_details = [(l.name, l._build_input_shape.dims[1].value) for l in self.model.layers]
        #print(self.model.summary())
        with open(
                f'{LSTM_MODEL_RESULTS}/{self.experiment_name}_input_window_{self.n_steps}_input_averaging_{self.input_averaging}_output_window{self.output_size}_output_averaging_{self.output_averaging}_modelsummary.txt',
                # _architecture_{self.architecture_name}
                'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def test(self, X_test, Y_test, normalization_arrays):
        print("X_test_shape_before_reshape", X_test.shape)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.n_features))
        print("X_test_shape_after_reshape", X_test.shape)
        # mse_test_score = self.model.evaluate(X_test, Y_test, batch_size=self.batch_size, verbose=0)
        Y_predict = self.model.predict(X_test)
        #print("x_test before denormalisation: ", X_test)
        #print("y_test before denormalisation: ", Y_test)
        #print("y_predict before denormalisation: ", Y_predict)
        # denormalize values
        _, x_min_test, _, x_max_test, _, company_name_test, _, input_period_start_date_test, _, input_period_end_date_test, _, output_period_start_date_test, _, output_period_end_date_test = normalization_arrays
        #_, x_min_test, _, x_max_test, _, _, _, _, _, _, _, _, _, _ = normalization_arrays
        x_min_test = np.array(x_min_test)
        x_max_test = np.array(x_max_test)
        print("x_min_test", x_min_test)
        print("x_max_test", x_max_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        print("X_test_shape_after_returning from reshape", X_test.shape)
        X_test = denormalize_x(array=X_test, x_min=x_min_test, x_max=x_max_test)
        Y_test = denormalize_y(array=Y_test, x_min=x_min_test, x_max=x_max_test)
        Y_predict = denormalize_y(array=Y_predict, x_min=x_min_test, x_max=x_max_test)
        # # calculate errors
        # Y_test = np.array(Y_test)
        # Y_predict = np.array(Y_predict)
        # print("y_test after denormalisation: ", Y_test)
        # print("y_predict after denormalisation: ", Y_predict)
        if self.use_log:
            print("reversing log operation (using exponent)")
            X_test = np.exp(X_test)
            Y_test = np.exp(Y_test)
            Y_predict = np.exp(Y_predict)
        # show predictions plot
        self.save_predictions(X_test, Y_test, Y_predict, company_name_test, input_period_start_date_test,
                              input_period_end_date_test, output_period_start_date_test, output_period_end_date_test)
        # save_results
        self.save_results(X_test, Y_test, Y_predict, company_name_test, input_period_start_date_test, input_period_end_date_test, output_period_start_date_test, output_period_end_date_test)
        # calculate errors
        mse_test_score = ((Y_test - Y_predict) ** 2).mean()
        mape_test_score = (abs((Y_test - Y_predict) / Y_test)).mean() * 100
        return mse_test_score, mape_test_score

    def run(self, X_train, Y_train, X_test, Y_test, normalization_arrays, epochs):
        # measure execution time
        start_time = datetime.now()
        self.epochs = epochs
        self.fit(X_train, Y_train, epochs=epochs)
        end_time = datetime.now()
        execution_time = end_time - start_time
        # history = self.results.history
        # calculate train errors
        mse_train_score = self.results.history['mse'][-1]
        mape_train_score = self.results.history['mape'][-1]
        mse_val_score = self.results.history['val_mse'][-1]
        mape_val_score = self.results.history['val_mape'][-1]
        # test errors
        mse_test_score, mape_test_score = self.test(X_test, Y_test, normalization_arrays)
        print(f"MSE TEST SCORE: {mse_test_score}")
        print(f"MSE VALIDATION SCORE: {mse_val_score}")
        print(f"MSE TRAIN SCORE: {mse_train_score}")
        print(f"MAPE TEST SCORE: {mape_test_score}")
        print(f"MAPE VALIDATION SCORE: {mape_val_score}")
        print(f"MAPE TRAIN SCORE: {mape_train_score}")
        #plot train errors
        self.plot_history()
        #Save experiment to MASTER TABLE
        self.save_experiment(mse_test_score=mse_test_score, mape_test_score=mape_test_score,
                             mse_train_score=mse_train_score, mape_train_score=mape_train_score,
                             mse_val_score=mse_val_score, mape_val_score=mape_val_score, exec_time=execution_time)

    def plot_history(self):
        history = self.results.history
        fig = plt.figure(figsize=(16, 5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title(f'Loss(mse) for {self.experiment_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        date_string = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        # TODO: date and plot id !!!!
        plt.savefig(
            f"{LSTM_PLOTS_PATH}/{self.experiment_name}_nn_training_history_mse_input_window_{self.n_steps}_input_averaging_{self.input_averaging}_output_window{self.output_size}_output_averaging_{self.output_averaging}_{date_string}.png")  # _{self.architecture_name}
        #plt.show()
        plt.close(fig)
        ###
        fig2 = plt.figure(figsize=(16, 5))
        plt.plot(history['val_mape'])
        plt.plot(history['mape'])
        plt.legend(['val_mape', 'mape'])
        plt.title(f'MAPE for {self.experiment_name}')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE')
        plt.savefig(
            f"{LSTM_PLOTS_PATH}/{self.experiment_name}_nn_training_history_mape_input_window_{self.n_steps}_input_averaging_{self.input_averaging}_output_window{self.output_size}_output_averaging_{self.output_averaging}_{date_string}.png")  # _{self.architecture_name}
        #plt.show()
        plt.close(fig2)

    def save_predictions(self, X_test, Y_test, Y_predict, company_name_test, input_period_start_date_test,
                         input_period_end_date_test, output_period_start_date_test, output_period_end_date_test):
        imageid = 0
        for i in range(len(Y_test)):  # len of X_test or Y_predict can be also used here
            imageid += 1
            company = company_name_test[i]

            input_start_date = input_period_start_date_test[i]
            input_end_date = input_period_end_date_test[i]
            output_start_date = output_period_start_date_test[i]
            output_end_date = output_period_end_date_test[i]

            # for clarity - change input_end_date in input_period_range to output_start_date
            if self.input_averaging:
                input_per_range = []
                input_per_range.append(pd.Timestamp(input_start_date)) #here first value is added
                for period in range(self.n_steps): #we start from 0 to n_steps-1, but we add +1 because of chart (the last date from input set + first from the output set)
                    input_per_range.append(pd.Timestamp(input_start_date) + DateOffset(days=30*(period+1)))
                input_period_range = pd.DatetimeIndex(input_per_range)
            else:
                input_period_range = pd.date_range(input_start_date, output_start_date)

            if self.output_averaging:
                output_per_range = []
                output_per_range.append(pd.Timestamp(output_start_date)) #here first value is added
                for period in range(self.output_size-1): # -1 because we have added already first value
                    output_per_range.append(pd.Timestamp(output_start_date) + DateOffset(days=30 * (period + 1)))
                output_period_range = pd.DatetimeIndex(output_per_range)
            else:
                output_period_range = pd.date_range(output_start_date, output_end_date)

            assert len(input_period_range) == (self.n_steps + 1), 'wrong input period len'
            assert len(output_period_range) == (self.output_size), 'wrong output period len'

            # temporary condition
            # if len(input_period_range) > self.input_window_size:
            #    input_period_range = input_period_range[:self.input_window_size]
            # if len(output_period_range) > self.output_window_size:
            #    output_period_range = output_period_range[:self.output_window_size]
            # x ranges
            # X_test_len = range(len(X_test[i]))
            # Y_test_len = range(X_test_len[-1], X_test_len[-1] + len(Y_test[i]))
            # Y_predict_len = range(X_test_len[-1], X_test_len[-1] + len(Y_predict[i]))
            fig = plt.figure(figsize=(14, 8))
            plt.title(f"{self.experiment_name} {company} company")
            plt.ylabel("Company price [\$]")
            plt.xlabel("Time periods")

            # for better plot - fill the hole + period tak samo
            tmp = np.append(X_test[i], Y_test[i][0])
            # input_period_range = ^up (u gory)

            plt.plot(input_period_range, tmp, "b-", label='training')  # "bo-" #X_test[i]
            plt.plot(output_period_range, Y_test[i], "g-", label='actual')  # "go-"
            plt.plot(output_period_range, Y_predict[i], "r-", label='forecast')  # "ro-"
            plt.legend()
            # plt.ylim(min,max) #zakres y od min do max
            date_string = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
            plt.savefig(
                f"{LSTM_PLOTS_PATH}/{self.experiment_name}_prediction_plot_id_{imageid}_input_window_{self.n_steps}_input_averaging_{self.input_averaging}_output_window{self.output_size}_output_averaging_{self.output_averaging}_{date_string}.png")  # _{self.architecture_name}
            # plt.show()
            plt.close(fig)
            # print just 5/10 plots
            # TODO: maybe change this condition
            if i > 20:
                break

    def save_results(self, X_test, Y_test, Y_predict, company_name_test, input_period_start_date_test, input_period_end_date_test,
                     output_period_start_date_test, output_period_end_date_test):
        # 1. Save experiment results
        df = pd.DataFrame()
        data_tuples = list(zip(X_test, Y_test, Y_predict, company_name_test, input_period_start_date_test, input_period_end_date_test,
                               output_period_start_date_test, output_period_end_date_test))
        df = df.append(data_tuples, ignore_index=True)
        df.columns = ['x_test', 'y_test', 'y_predict', 'company_name', 'x_start_date', 'x_end_date', 'y_start_date',
                      'y_end_date']
        # print(df)
        date_string = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        df.to_csv(f"{LSTM_MODEL_RESULTS}/{self.experiment_name}_results_{date_string}.csv")
        # 2. Save loss and val_loss values
        df2 = pd.DataFrame()
        data_tuples2 = list(zip(self.results.history['mse'], self.results.history['val_mse'],
                                self.results.history['mape'], self.results.history['val_mape']))
        df2 = df2.append(data_tuples2, ignore_index=True)
        df2.columns = ['mse_train', 'mse_val', 'mape_train', 'mape_val']
        df2.to_csv(
            f"{LSTM_MODEL_RESULTS}/{self.experiment_name}_errors_input_window_{self.n_steps}_input_averaging_{self.input_averaging}_output_window{self.output_size}_output_averaging_{self.output_averaging}_{date_string}.csv")

    def save_experiment(self, mse_test_score, mape_test_score, mse_train_score, mape_train_score,
                        mse_val_score, mape_val_score, exec_time):
        MASTER_COLUMN_NAMES = ['timestamp', 'experiment_id', 'experiment_desc', 'model_name',  # 'model_details',
                               'optimizer', 'activation_func', 'epochs',
                               'input_size', 'output_size', 'input_averaging', 'output_averaging', 'use_log',
                               'MSE_test', 'MSE_val', 'MSE_train',
                               'MAPE_test', 'MAPE_val', 'MAPE_train', 'exec_runtime_hours', 'exec_runtime']
        master_table_df = pd.DataFrame(columns=MASTER_COLUMN_NAMES)
        exec_hours = exec_time.seconds // 3600
        exec_minutes = (exec_time.seconds % 3600) // 60
        exec_seconds = (exec_time.seconds % 3600) % 60
        exec_time_h = f"{exec_hours}h {exec_minutes}min {exec_seconds}s"
        print(f"Duration of model training: {exec_time_h}")
        master_table_values = [datetime.now().strftime('%Y-%m-%d_%H_%M'),
                               self.experiment_id,
                               self.experiment_name,
                               self.architecture_name,
                               # self.model_details,
                               self.optimizer_name,
                               self.activation,
                               self.epochs,
                               self.n_steps,
                               self.output_size,
                               self.input_averaging,
                               self.output_averaging,
                               self.use_log,
                               mse_test_score, mse_val_score, mse_train_score,
                               mape_test_score, mape_val_score, mape_train_score,
                               exec_time_h,
                               exec_time]
        master_table_df = master_table_df.append(pd.Series(master_table_values, index=master_table_df.columns),
                                                 ignore_index=True)
        # SAVE MASTER TABLE WITH ALL THE DATA
        print(master_table_df)
        master_file_name = LSTM_MODEL_RESULTS + f'/MASTER_TABLE.csv'
        file_exists = os.path.isfile(master_file_name)
        if not file_exists:
            with open(master_file_name, 'x') as f:
                master_table_df.to_csv(master_file_name)
                f.close()
        else:
            master_table_df.to_csv(master_file_name, mode='a', header=None)

    # TODO: może kiedyś jak nie będzie działać
    # def reset_model_weights(self):
    #     session = K.get_session()
    #     for layer in model.layers:
    #         for v in layer.__dict__:
    #             v_arg = getattr(layer, v)
    #             if hasattr(v_arg, 'initializer'):
    #                 initializer_method = getattr(v_arg, 'initializer')
    #                 initializer_method.run(session=session)
    #                 print('reinitializing layer {}.{}'.format(layer.name, v))
