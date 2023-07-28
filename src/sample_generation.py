import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.constants import CLEANED_DATA_PATHS, SAMPLES_PATH, ARIMA_SAMPLES_PATH, PREPROCESSING_PLOTS_PATH, \
    ARIMA_PLOTS_PATH, \
    SAMPLES_PLOTS_PATH
from src.utils import check_if_sample_exists, unpickle_array, pickle_array, check_if_series_sample_exists
from sklearn.model_selection import train_test_split

# set global random_seed
np.random.seed(7)


class ARIMASeriesGenerator:
    def __init__(self, series_averaging=False, use_log=True, save=True):
        self.series_averaging = series_averaging
        self.data_paths = CLEANED_DATA_PATHS
        self.use_log = use_log
        self.test_size = 0.33
        self.save = save

    def generate_series(self):
        # tmp_counter = 0
        for path in tqdm(self.data_paths, desc='Generating Series Samples'):
            # extract company name for further plots
            splitted = path.split("\\")[-1]
            company = splitted.split(".csv")[0]
            series_name = self._generate_series_name(company)
            if check_if_series_sample_exists(series_name=series_name) == False:
                print("generating series")
                series = self._generate_series(company, path)
                # visualization
                self.visualize_series(series, company)
                # log and its visulisation
                if self.use_log:
                    series.avg_price = np.log(series.avg_price)
                    self.visualize_after_log(series, company)
                if self.save:
                    self._save_series(series, series_name=series_name)
            # temporary constraint for dataset
            # delete tmp constraint
            # tmp_counter += 1
            # if tmp_counter == 10:
            #    break

    def _generate_series(self, company, path):
        # read_csv
        df = pd.read_csv(path, index_col=0)
        df.timestamp = pd.to_datetime(df.timestamp)
        df.sort_values(by='timestamp', inplace=True)
        series = df
        # check frequency
        if self.series_averaging:
            series = self._series_averaging(series)
        # else:
        #    series.set_index('timestamp', inplace=True)
        return series

    def _series_averaging(self, series):
        df = series.groupby(pd.Grouper(key='timestamp', freq='1M')).agg(
            avg_price=('avg_price', 'mean')
        )
        # df.index = df.index.strftime('%Y-%m') #optional change of formatting
        df.insert(0, 'timestamp', df.index)
        df.index = range(0, len(df))
        return df

    def _generate_series_name(self, company):
        return f"{company}_series_averaging_{self.series_averaging}_use_log_{self.use_log}"

    # def _load_samples(self, series_name):
    #    file_path = f'{SAMPLES_PATH}/{series_name}.pickle'
    #    df = pd.read_csv(file_path)
    #    return df

    def _save_series(self, series, series_name):
        new_file_path = ARIMA_SAMPLES_PATH + '\\' + series_name + '.csv'
        series.to_csv(new_file_path)

    def visualize_series(self, df, company):
        if self.series_averaging:
            interval = 'monthly'
        else:
            interval = 'daily'
        # time interval for the given company
        starting_year = df.timestamp[0].year
        ending_year = df.timestamp[len(df) - 1].year
        # visualize
        plt.style.use('fivethirtyeight')
        plt.style.use('seaborn-bright')
        fig = plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(df['timestamp'], df['avg_price'], linewidth=2, linestyle='solid', color='black')
        plt.title('{0} stock {1} average prices {2} - {3}'.format(company, interval, starting_year, ending_year))
        plt.xlabel('Date')
        plt.ylabel('{0} average price values [\$]'.format(interval))
        plt.legend(['{0}'.format(company)])
        plt.savefig(
            SAMPLES_PLOTS_PATH + '/{0}_stock_{1}_average_prices_{2}_{3}'.format(company, interval, starting_year,
                                                                                ending_year))
        # plt.show()
        plt.close(fig)

    def visualize_after_log(self, df, company):
        if self.series_averaging:
            interval = 'monthly'
        else:
            interval = 'daily'
        fig = plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(df['timestamp'], df['avg_price'], color='purple', linewidth=2, linestyle='solid')
        plt.title("{0} after logarithm".format(company))
        plt.xlabel('Date')
        plt.ylabel('Daily average log price values [log \$]')
        plt.legend(['{0}'.format(company)])
        plt.savefig(SAMPLES_PLOTS_PATH + '/{0}_{1}_after_log.png'.format(company, interval));
        # plt.show()
        plt.close(fig)


class SampleGenerator:
    def __init__(self, input_window_size, output_window_size, freq=30,
                 output_averaging=False, input_averaging=False, use_log=False, save=True):
        self.shift = freq  # TODO jeśli za mało danych to zmienić na częstsze: 3dni, tydzień etc.
        self.X = None
        self.Y = None
        self.test_size = 0.33
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.freq = freq
        self.output_averaging = output_averaging
        self.input_averaging = input_averaging
        self.use_log = use_log
        self.save = save
        self.data_paths = CLEANED_DATA_PATHS

    def generate(self, split=False):
        sample_name = self._generate_sample_name()
        print("sample_name: ", sample_name)
        if check_if_sample_exists(sample_name=sample_name):
            print(f'Loading sample from backup: {sample_name}')
            self._load_samples(sample_name=sample_name)
            print(f'Loading normalisation from backup: {sample_name}')
            x_min_list, x_max_list, company_name_list, input_period_start_date_list, input_period_end_date_list, \
            output_period_start_date_list, output_period_end_date_list = self._load_normalisation(
                sample_name=sample_name)
        else:
            print("generating sample: \n")
            x_min_list, x_max_list, company_name_list, input_period_start_date_list, input_period_end_date_list, \
            output_period_start_date_list, output_period_end_date_list = self._generate()
            print("sample generated\n")
            if self.save:
                print("saving sample: \n")
                self._save_sample(sample_name=sample_name)
                print("saving normalisation: \n")
                self._save_normalization(x_min_list, x_max_list, company_name_list, input_period_start_date_list,
                                         input_period_end_date_list, \
                                         output_period_start_date_list, output_period_end_date_list,
                                         sample_name=sample_name)
                print("normalisation saved")
        # GENERATE HISTOGRAMS
        # self.X
        # self.Y
        #
        if split:
            print("performing split: \n")
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size,
                                                                random_state=7)
            x_min_train, x_min_test = train_test_split(x_min_list, test_size=self.test_size, random_state=7)
            x_max_train, x_max_test = train_test_split(x_max_list, test_size=self.test_size, random_state=7)
            company_name_train, company_name_test = train_test_split(company_name_list, test_size=self.test_size,
                                                                     random_state=7)
            input_period_start_date_train, input_period_start_date_test = train_test_split(input_period_start_date_list,
                                                                                           test_size=self.test_size,
                                                                                           random_state=7)
            input_period_end_date_train, input_period_end_date_test = train_test_split(input_period_end_date_list,
                                                                                       test_size=self.test_size,
                                                                                       random_state=7)
            output_period_start_date_train, output_period_start_date_test = train_test_split(
                output_period_start_date_list, test_size=self.test_size, random_state=7)
            output_period_end_date_train, output_period_end_date_test = train_test_split(output_period_end_date_list,
                                                                                         test_size=self.test_size,
                                                                                         random_state=7)
            return X_train, X_test, Y_train, Y_test, (x_min_train, x_min_test, x_max_train, x_max_test,
                                                      company_name_train, company_name_test,
                                                      input_period_start_date_train, input_period_start_date_test,
                                                      input_period_end_date_train, input_period_end_date_test,
                                                      output_period_start_date_train, output_period_start_date_test,
                                                      output_period_end_date_train, output_period_end_date_test)
        else:
            return self.X, self.Y, x_min_list, x_max_list, company_name_list, input_period_start_date_list, input_period_end_date_list, \
                   output_period_start_date_list, output_period_end_date_list

    def _generate(self):
        # lists with normalisations
        company_name_list = []
        input_period_start_date_list = []
        input_period_end_date_list = []
        output_period_start_date_list = []
        output_period_end_date_list = []
        x_min_list = []
        x_max_list = []
        X_df = pd.DataFrame()
        Y_df = pd.DataFrame()
        for path in tqdm(self.data_paths, desc='Generating Samples'):
            company_name = path.split('../data/5_cleaned_data\\')[1].replace(".csv", "")
            df = pd.read_csv(path, index_col=0).reset_index(drop=True)
            df.timestamp = pd.to_datetime(df.timestamp)
            df.sort_values(by='timestamp', inplace=True)
            if self.use_log:
                df.avg_price = np.log(df.avg_price)
            for i in range(0, df.shape[0] - self.input_window_size - self.output_window_size, self.shift):

                input_period_df = df.loc[i:i + self.input_window_size - 1, :].reset_index(drop=True)
                input_period = input_period_df.avg_price
                input_period_start_date = input_period_df.timestamp.min()
                input_period_end_date = input_period_df.timestamp.max()

                output_period_df = df.loc[
                                   i + self.input_window_size: i + self.input_window_size + self.output_window_size - 1,
                                   :].reset_index(drop=True)
                output_period = output_period_df.avg_price
                output_period_start_date = output_period_df.timestamp.min()
                output_period_end_date = output_period_df.timestamp.max()

                input_period_min = input_period.min()
                input_period_max = input_period.max()
                input_period = (input_period - input_period_min) / (input_period_max - input_period_min)
                output_period = (output_period - input_period_min) / (input_period_max - input_period_min)

                # we are testing different periods
                # if len(input_period) != 90:
                #    print("dupa", company_name, i, df.shape[0])
                # if len(output_period) != 30:
                #    print("dupa", company_name, i, df.shape[0])

                company_name_list.append(company_name)
                input_period_start_date_list.append(input_period_start_date)
                input_period_end_date_list.append(input_period_end_date)
                output_period_start_date_list.append(output_period_start_date)
                output_period_end_date_list.append(output_period_end_date)

                # Check
                input_period_days_span = (input_period_end_date - input_period_start_date).days
                output_period_days_span = (output_period_end_date - output_period_start_date).days

                assert input_period_days_span == self.input_window_size - 1, \
                    f"Input size mismatch {input_period_days_span} != {self.input_window_size - 1}"
                assert output_period_days_span == self.output_window_size - 1, \
                    f"Output size mismatch {output_period_days_span} != {self.output_window_size - 1}"

                x_min_list.append(input_period_min)
                x_max_list.append(input_period_max)

                if self.output_averaging:
                    output_period = self._series_averaging(output_period)
                if self.input_averaging:
                    input_period = self._series_averaging(input_period)

                X_df = X_df.append(input_period)
                Y_df = Y_df.append(output_period)
        X_df = X_df.reset_index(drop=True)
        Y_df = Y_df.reset_index(drop=True)
        self.X = X_df.values
        self.Y = Y_df.values
        assert np.isnan(self.X).sum() == 0, 'Nans in X array'
        assert np.isnan(self.Y).sum() == 0, 'Nans in Y array'
        return x_min_list, x_max_list, company_name_list, input_period_start_date_list, input_period_end_date_list, \
               output_period_start_date_list, output_period_end_date_list

    def _series_averaging(self, series):
        """
        :param divide dataframe records into classes; number of classes depends on frequency:
        :return: average price value for every class
        """
        series = series.reset_index(drop=True)
        _series = pd.DataFrame(series)
        _series['group'] = series.index // self.freq
        return _series.groupby('group').mean().avg_price

    def _generate_sample_name(self):
        return f"input_window_size_{self.input_window_size}_output_window_size_{self.output_window_size}_freq_{self.freq}_input_averaging_{self.input_averaging}_output_averaging_{self.output_averaging}_use_log_{self.use_log}"

    def _load_samples(self, sample_name):
        X_file_path = f'{SAMPLES_PATH}/X_{sample_name}.pickle'
        Y_file_path = f'{SAMPLES_PATH}/Y_{sample_name}.pickle'
        self.X = unpickle_array(file_name=X_file_path)
        self.Y = unpickle_array(file_name=Y_file_path)

    def _save_sample(self, sample_name):
        X_file_path = f'{SAMPLES_PATH}/X_{sample_name}.pickle'
        Y_file_path = f'{SAMPLES_PATH}/Y_{sample_name}.pickle'
        pickle_array(array=self.X, file_name=X_file_path)
        pickle_array(array=self.Y, file_name=Y_file_path)
        print("sample", sample_name, "saved\n")

    def _save_normalization(self, x_min_list, x_max_list, company_name_list, input_period_start_date_list,
                            input_period_end_date_list, \
                            output_period_start_date_list, output_period_end_date_list, sample_name):
        data = {'x_min': x_min_list,
                'x_max': x_max_list,
                'company_name': company_name_list,
                'input_period_start_date': input_period_start_date_list,
                'input_period_end_date': input_period_end_date_list,
                'output_period_start_date': output_period_start_date_list,
                'output_period_end_date': output_period_end_date_list}
        df = pd.DataFrame(data)
        df.to_csv(f'{SAMPLES_PATH}/normalisation_{sample_name}.csv')
        print("df normalisation", df)

    def _load_normalisation(self, sample_name):
        df = pd.read_csv(f'{SAMPLES_PATH}/normalisation_{sample_name}.csv')
        x_min_list = df['x_min'].values
        x_max_list = df['x_max'].values
        company_name_list = df['company_name'].values
        input_period_start_date_list = df['input_period_start_date'].values
        input_period_end_date_list = df['input_period_end_date'].values
        output_period_start_date_list = df['output_period_start_date'].values
        output_period_end_date_list = df['output_period_end_date'].values
        return x_min_list, x_max_list, company_name_list, input_period_start_date_list, input_period_end_date_list, \
               output_period_start_date_list, output_period_end_date_list


if __name__ == '__main__':
    sg = SampleGenerator(
        input_window_size=120,
        output_window_size=90,
        output_averaging=True,
        input_averaging=True,
        save=False
    )
    X, Y = sg.generate()
