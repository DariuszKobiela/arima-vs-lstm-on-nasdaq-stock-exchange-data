import os

import warnings

from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.python.keras.layers import GRU, Dense, LSTM, RNN

from src.models import NeuralNetworkModel, ARIMAModel
from src.sample_generation import SampleGenerator, ARIMASeriesGenerator

# set global random seed
import tensorflow as tf

tf.random.set_seed(7)
warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)

if __name__ == '__main__':
    # TODO: maybe later create histogram after sample generation
    # TODO: maybe later model_name dodać liczbę unitów; nie działa dla Dense - porzucam, zbyt czasochlonne
    #  [(l.name, l._build_input_shape.dims[1].value) for l in model.layers]

    #
    # sg = SampleGenerator(
    #     input_window_size=90,
    #     output_window_size=30,
    # )
    # X, Y = sg.generate()
    ###
    # Ctrl+'/' - comment/uncomment section
    """ 
        Experiment 1 WORKING
        Predict [7, 14, 30, 60] days in ARIMA and choose p, q ranges
    """
    # experiment_name = "Experiment_1_Predict_[7,14,30,60]_days_in_ARIMA_and_choose_p_q_ranges"
    # experiment_id = 1
    # output_window_size = [7, 14]  # always a list [7, 14, 30, 60]
    # p_range = range(0, 1)  # list of possible p paramaters, range(0, 10)
    # q_range = range(0, 1)  # list of possible q paramaters, range(0, 10)
    # P_VALUE_PERCENT = '5%'
    # ARIMASeriesGenerator(
    #     series_averaging=False,
    #     use_log=True,
    #     save=True
    # ).generate_series()
    #
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()

    """ 
       Experiment 2
       architecture comparison LSTM
    """
    # # default activation function is "tanh"
    # # LSTM(30, activation="tanh")
    # experiment_name = "Experiment_2_architecture_comparison_LSTM"
    # architecture_name = ["LSTM+Dense", "2xLSTM+Dense", "2xLSTM", "3xLSTM"]
    # experiment_id = 2
    # use_log = True
    # input_window_size = 90
    # output_window_size = 30
    # epochs = 200
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=False,
    #     input_averaging=False,
    #     save=True
    # ).generate(split=True)
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=False,
    #     output_averaging=False,
    #     optimizer=Adam(),
    #     optimizer_name='Adam',
    #     activation='tanh',
    #     use_log=use_log
    # )
    # architectures = [
    #     [
    #         LSTM(30),
    #         Dense(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(60, return_sequences=True),
    #         LSTM(30),
    #         Dense(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(60, return_sequences=True),
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ]
    # ]
    #
    # for i in range(len(architectures)):
    #     nn_model.prepare(layers_list=architectures[i], architecture_name=architecture_name[i])
    #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)
    """
       Experiment 3
       layers comparison GRU LSTM
    """
    # experiment_name = "Experiment_3_layers_comparison_GRU_LSTM"
    # experiment_id = 3
    # architecture_name = ["2xGRU", "3xGRU", "2xLSTM", "3xLSTM"]
    # use_log = True
    # input_window_size = 90
    # output_window_size = 30
    # epochs = 250
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=False,
    #     input_averaging=False,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=False,
    #     output_averaging=False,
    #     optimizer=Adam(),
    #     optimizer_name='Adam',
    #     activation='tanh',
    #     use_log=use_log
    # )
    # architectures = [
    #     # It is not possible to use RNN in such way
    #     # [
    #     #     RNN(30, return_sequences=True),
    #     #     RNN(nn_model.output_size),
    #     # ],
    #     [
    #         GRU(30, return_sequences=True),
    #         GRU(nn_model.output_size),
    #     ],
    #     [
    #         GRU(60, return_sequences=True),
    #         GRU(30, return_sequences=True),
    #         GRU(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(60, return_sequences=True),
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ]
    # ]
    #
    # for i in range(len(architectures)):
    #     nn_model.prepare(layers_list=architectures[i], architecture_name=architecture_name[i])
    #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """
      Experiment 4
      testing hiperparameters LSTM
    """
    # experiment_name = "Experiment_4_testing_hiperparameters_LSTM"
    # experiment_id = 4
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_window_size = 90
    # output_window_size = 30
    # epochs = 250
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=False,
    #     input_averaging=False,
    #     save=True
    # ).generate(split=True)
    #
    # #optimizers = [RMSprop(), Adadelta(), Adagrad(), Adam(), SGD()]
    # #optimizer_names = ['RMSprop', 'Adadelta', 'Adagrad', 'Adam', 'SGD']
    # activation_functions = ['linear', 'relu', 'sigmoid', 'tanh']
    # # Czy po prostu 200 i Early stopping? M: daj 200 i early stopping
    # # default LSTM activation="tanh", recurrent_activation=deafault i nie badamy
    # # validation_split = 0.2 ?  chyba 0.3 był spoko
    # # EarlyStopping patience=25?  tak ale to trzeba patrzeć na wykresy czy zebigają i się wypłaszczają, bo jak leci w dół to go nie stopujmmy
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=False,
    #     output_averaging=False,
    #     optimizer=Adadelta(),
    #     optimizer_name='Adadelta',
    #     activation='tanh',
    #     use_log=use_log
    # )
    #
    # # architecture = [
    # #     LSTM(30, return_sequences=True),
    # #     LSTM(nn_model.output_size),
    # # ]
    #
    # # for i in range(len(optimizers)):
    # #     nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, optimizer=optimizers[i], optimizer_name=optimizer_names[i])
    # #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)
    #
    # for activation in activation_functions:
    #     architecture = [
    #         LSTM(30, return_sequences=True, activation=activation),
    #         LSTM(nn_model.output_size, activation=activation)
    #     ]
    #     nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation)
    #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """
      Experiment 5
      input window size
    """
    # experiment_name = "Experiment_5_input_window_size_LSTM"
    # experiment_id = 5
    # architecture_name = "2xLSTM"
    # use_log = True
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_sizes = [60, 90, 120, 150, 180, 240, 360, 480, 720]
    # output_window_size = 30
    # epochs = 250
    #
    # for input_window_size in input_window_sizes:
    #     X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #         input_window_size=input_window_size,
    #         output_window_size=output_window_size,
    #         use_log=use_log,
    #         output_averaging=False,
    #         input_averaging=False,
    #         save=True
    #     ).generate(split=True)
    #
    #     nn_model = NeuralNetworkModel(
    #         experiment_name=experiment_name,
    #         experiment_id=experiment_id,
    #         input_window_size=input_window_size,
    #         output_window_size=output_window_size,
    #         input_averaging=False,
    #         output_averaging=False,
    #         optimizer=optimizer,
    #         optimizer_name=optimizer_name,
    #         activation=activation_function,
    #         use_log=use_log
    #     )
    #
    #     architecture = [
    #         LSTM(30, return_sequences=True, activation=activation_function),
    #         LSTM(nn_model.output_size, activation=activation_function)
    #     ]
    #     nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function, optimizer=optimizer, optimizer_name=optimizer_name)
    #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """
      Experiment 6
      input window size with averaging
    """
    # experiment_name = "Experiment_6_input_window_size_with_averaging"
    # experiment_id = 6
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = True
    # output_averaging = True
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_sizes = [12*30, 24*30, 60*30, 90*30]
    # output_window_size = 90
    # epochs = 250
    #
    # for input_window_size in input_window_sizes:
    #     X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #         input_window_size=input_window_size,
    #         output_window_size=output_window_size,
    #         use_log=use_log,
    #         output_averaging=output_averaging,
    #         input_averaging=input_averaging,
    #         save=True
    #     ).generate(split=True)
    #
    #     nn_model = NeuralNetworkModel(
    #         experiment_name=experiment_name,
    #         experiment_id=experiment_id,
    #         input_window_size=input_window_size,
    #         output_window_size=output_window_size,
    #         input_averaging=input_averaging,
    #         output_averaging=output_averaging,
    #         optimizer=optimizer,
    #         optimizer_name=optimizer_name,
    #         activation=activation_function,
    #         use_log=use_log
    #     )
    #
    #     architecture = [
    #         LSTM(30, return_sequences=True, activation=activation_function),
    #         LSTM(nn_model.output_size, activation=activation_function)
    #     ]
    #     nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                      optimizer=optimizer, optimizer_name=optimizer_name)
    #     nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)
    """
        Experiment 7
        Predict 1 day in ARIMA
    """
    # experiment_name = "Experiment_7_Predict_1_day_ARIMA"
    # experiment_id = 7
    # output_window_size = [1]  # always a list [7, 14, 30, 60]
    # p_range = range(0, 3)  # list of possible p paramaters
    # q_range = range(0, 3)  # list of possible q paramaters
    # P_VALUE_PERCENT = '5%'
    # ARIMASeriesGenerator(
    #     series_averaging=False,
    #     use_log=True,
    #     save=True
    # ).generate_series()
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()
    """
        Experiment 8
        Predict 1 day in LSTM
    """
    # experiment_name = "Experiment_8_Predict_1_day_LSTM"
    # experiment_id = 8
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = False
    # output_averaging = False
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_size = 180
    # output_window_size = 1
    # epochs = 250
    #
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=output_averaging,
    #     input_averaging=input_averaging,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=input_averaging,
    #     output_averaging=output_averaging,
    #     optimizer=optimizer,
    #     optimizer_name=optimizer_name,
    #     activation=activation_function,
    #     use_log=use_log
    # )
    #
    # architecture = [
    #     LSTM(30, return_sequences=True, activation=activation_function),
    #     LSTM(nn_model.output_size, activation=activation_function)
    # ]
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                  optimizer=optimizer, optimizer_name=optimizer_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)
    """
        Experiment 9
        Predict 30 days in ARIMA
    """
    # experiment_name = "Experiment_9_Predict_30_days_ARIMA"
    # experiment_id = 9
    # output_window_size = [30]  # always a list [7, 14, 30, 60]
    # p_range = range(0, 3)  # list of possible p paramaters
    # q_range = range(0, 3)  # list of possible q paramaters
    # P_VALUE_PERCENT = '5%'
    # ARIMASeriesGenerator(
    #     series_averaging=False,
    #     use_log=True,
    #     save=True
    # ).generate_series()
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()

    """
        Experiment 10
        Predict 30 days in LSTM
    """
    # experiment_name = "Experiment_10_Predict_30_days_LSTM"
    # experiment_id = 10
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = False
    # output_averaging = False
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_size = 180
    # output_window_size = 30
    # epochs = 250
    #
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=output_averaging,
    #     input_averaging=input_averaging,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=input_averaging,
    #     output_averaging=output_averaging,
    #     optimizer=optimizer,
    #     optimizer_name=optimizer_name,
    #     activation=activation_function,
    #     use_log=use_log
    # )
    #
    # architecture = [
    #     LSTM(30, return_sequences=True, activation=activation_function),
    #     LSTM(nn_model.output_size, activation=activation_function)
    # ]
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                  optimizer=optimizer, optimizer_name=optimizer_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """
        Experiment 11
        Predict 1 month in ARIMA
    """
    # # PERIOD=365 dla danych dzienych (tyle dni jest w roku), PERIOD=12 dla danych miesięcznych (tyle jest miesięcy w roku)
    # experiment_name = "Experiment_11_Predict_1_month_ARIMA"
    # experiment_id = 11
    # output_window_size = [1]  # always a list [7, 14, 30, 60]
    # p_range = range(0, 3)  # list of possible p paramaters
    # q_range = range(0, 3)  # list of possible q paramaters
    # P_VALUE_PERCENT = '5%'
    # series_averaging = True
    #
    # ARIMASeriesGenerator(
    #     series_averaging=series_averaging,
    #     use_log=True,
    #     save=True
    # ).generate_series()
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     series_averaging=series_averaging,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT,
    #     PERIOD=12
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()

    """
        Experiment 12
        Predict 1 month in LSTM
    """
    # experiment_name = "Experiment_12_Predict_1_month_LSTM"
    # experiment_id = 12
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = True
    # output_averaging = True
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_size = 1800
    # output_window_size = 30
    # epochs = 250
    #
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=output_averaging,
    #     input_averaging=input_averaging,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=input_averaging,
    #     output_averaging=output_averaging,
    #     optimizer=optimizer,
    #     optimizer_name=optimizer_name,
    #     activation=activation_function,
    #     use_log=use_log
    # )
    # architecture = [
    #     LSTM(30, return_sequences=True, activation=activation_function),
    #     LSTM(nn_model.output_size, activation=activation_function)
    # ]
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                  optimizer=optimizer, optimizer_name=optimizer_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """ 
        Experiment 13 WORKING
        Predict [3] months in ARIMA (pq range wynika z eksperymentu 1)
    """
    # #PERIOD=365 dla danych dzienych (tyle dni jest w roku), PERIOD=12 dla danych miesięcznych (tyle jest miesięcy w roku)
    # experiment_name = "Experiment_13_Predict_3_months_in_ARIMA"
    # experiment_id = 13
    # output_window_size = [3] #always a list
    # p_range = range(0, 3) #list of possible p paramaters
    # q_range = range(0, 3) #list of possible q paramaters
    # P_VALUE_PERCENT = '5%'
    # series_averaging = True
    #
    # # ARIMASeriesGenerator(
    # #     series_averaging=series_averaging,
    # #     use_log=True,
    # #     save=True
    # # ).generate_series()
    #
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     series_averaging=series_averaging,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT,
    #     PERIOD=12
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()

    """
        Experiment 14
        Predict 3 months in LSTM
    """
    # experiment_name = "Experiment_14_Predict_3_months_LSTM"
    # experiment_id = 14
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = True
    # output_averaging = True
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_size = 1800
    # output_window_size = 90
    # epochs = 250
    #
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=output_averaging,
    #     input_averaging=input_averaging,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=input_averaging,
    #     output_averaging=output_averaging,
    #     optimizer=optimizer,
    #     optimizer_name=optimizer_name,
    #     activation=activation_function,
    #     use_log=use_log
    # )
    # architecture = [
    #     LSTM(30, return_sequences=True, activation=activation_function),
    #     LSTM(nn_model.output_size, activation=activation_function)
    # ]
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                  optimizer=optimizer, optimizer_name=optimizer_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)

    """ 
        Experiment 15 WORKING
        Predict [9] months in ARIMA 
    """
    # #PERIOD=365 dla danych dzienych (tyle dni jest w roku), PERIOD=12 dla danych miesięcznych (tyle jest miesięcy w roku)
    # experiment_name = "Experiment_15_Predict_9_months_in_ARIMA"
    # experiment_id = 15
    # output_window_size = [9] #always a list
    # p_range = range(0, 3) #list of possible p paramaters
    # q_range = range(0, 3) #list of possible q paramaters
    # P_VALUE_PERCENT = '5%'
    # series_averaging = True
    #
    # # ARIMASeriesGenerator(
    # #     series_averaging=series_averaging,
    # #     use_log=True,
    # #     save=True
    # # ).generate_series()
    #
    # arima_model = ARIMAModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     output_window_size=output_window_size,
    #     series_averaging=series_averaging,
    #     p_range=p_range,
    #     q_range=q_range,
    #     P_VALUE_PERCENT=P_VALUE_PERCENT,
    #     PERIOD=12
    # )
    # arima_model.ensure_stationarity_and_perform_forecast()

    """
    Experiment 16
    Predict 9 months in LSTM
    """
    # experiment_name = "Experiment_16_Predict_9_months_LSTM"
    # experiment_id = 16
    # architecture_name = "2xLSTM"
    # use_log = True
    # input_averaging = True
    # output_averaging = True
    # optimizer = Adam()
    # optimizer_name = 'Adam'
    # activation_function = 'tanh'
    # input_window_size = 1800
    # output_window_size = 270
    # epochs = 250
    #
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=output_averaging,
    #     input_averaging=input_averaging,
    #     save=True
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=input_averaging,
    #     output_averaging=output_averaging,
    #     optimizer=optimizer,
    #     optimizer_name=optimizer_name,
    #     activation=activation_function,
    #     use_log=use_log
    # )
    # architecture = [
    #     LSTM(30, return_sequences=True, activation=activation_function),
    #     LSTM(nn_model.output_size, activation=activation_function)
    # ]
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name, activation=activation_function,
    #                  optimizer=optimizer, optimizer_name=optimizer_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays, epochs=epochs)
    """
    END OF EXPERIMENTS
    """


    #EXTRA EXPERIMENTS
    """
    Experiment 0
    use log for LSTM
    """
    # experiment_name = "Experiment_0_use_log_LSTM"
    # experiment_id = 0
    # input_window_size = 90
    # output_window_size = 30
    # # for use_log in [True, False]:
    # use_log = True
    # architecture_name = "GRU+Dense"
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     experiment_id=experiment_id,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     input_averaging=False,
    #     output_averaging=False,
    #     optimizer=Adam(),
    #     optimizer_name='Adam',
    #     use_log=use_log
    # )
    # architecture = [
    #     GRU(10, return_sequences=False),
    #     Dense(nn_model.output_size),
    # ]
    # X_train, X_test, Y_train, Y_test, normalization_arrays = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log,
    #     output_averaging=False,
    #     input_averaging=False,
    #     save=True
    # ).generate(split=True)
    # nn_model.prepare(layers_list=architecture, architecture_name=architecture_name)
    # nn_model.run(X_train, Y_train, X_test, Y_test, normalization_arrays)
    """
    Experiment 3
    architecture comparison GRU
    """
    # experiment_name = "Experiment 3: architecture comparison GRU"
    # use_log = False
    # input_window_size = 90
    # output_window_size = 30
    # X_train, X_test, Y_train, Y_test = SampleGenerator(
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size,
    #     use_log=use_log
    # ).generate(split=True)
    #
    # nn_model = NeuralNetworkModel(
    #     experiment_name=experiment_name,
    #     input_window_size=input_window_size,
    #     output_window_size=output_window_size
    # )
    # architectures = [
    #     [
    #         LSTM(30),
    #         Dense(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(60, return_sequences=True),
    #         LSTM(30),
    #         Dense(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ],
    #     [
    #         LSTM(60, return_sequences=True),
    #         LSTM(30, return_sequences=True),
    #         LSTM(nn_model.output_size),
    #     ],
    # ]
    #
    # for architecture in architectures:
    #     nn_model.prepare(architecture)
    #     nn_model.run(X_train, Y_train, X_test, Y_test, use_log)
    #