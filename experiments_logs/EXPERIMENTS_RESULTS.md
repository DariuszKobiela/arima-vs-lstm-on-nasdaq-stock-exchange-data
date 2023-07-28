# METRICS USED:
- MSE is metric just for LSTM (because we observe loss functin, which is mse)
- MAPE is metric for ARIMA (because is is more accurate - absolute metric for companies with different sizes)
- MAPE is metric to compare LSTM with ARIMA (same order of size for both models). 

# COMPARISON BUSINESS SCENARIOS REASONS:
- We have decided to compare models on 3 different periods: 30 days, 3 months and 9 months.
- Reasons:
	- odzwieciedlenie róznych możliwości inwestycyjnych (krótko, średnio i długookresowe)
	- każdy ma swoje preferencje inwestycyjny: chcieliśmy, aby nasza praca miała sens biznesowy i aby mogła być użyteczna dla różnego typu inwestorów. 

- Additional 2 basis comparisons are performes: 1 day and 1 month, to check if LSTM or ARIMA performs better in case of prediction just 1 step ahead. 


# Experiment_1_Predict_[7,14,30,60]_days_in_ARIMA_and_choose_p_q_ranges
$ p = range(0,8)
$ q = range(0,8)
$ series_averaging=False
$ output_window_size = [7, 14, 30, 60]

- Whole experiment execution time: 4h 36min 03s (Gooogle COLAB)

- Mean MSE for different periods:
	- 7 = 96.407445
	- 14 = 71.899530
	- 30 = 61.523715
	- 60 = 49.313363
	
- Mean MAPE for different periods:
	- 7 = 1.184635
	- 14 = 1.242393
	- 30 = 1.133631
	- 60 = 1.093943
	
- Mean MSE for different models:
	*ARIMA(6, 1, 5)       0.020479
	*ARIMA(3, 1, 1)       0.026731
	*ARIMA(1, 1, 7)       0.036101
	*ARIMA(1, 0, 1)       0.077138
	*ARIMA(3, 1, 3)       0.183879
	*ARIMA(7, 1, 6)       0.334097
	*ARIMA(5, 1, 4)       0.611477
	*ARIMA(1, 0, 5)       0.657239
	*ARIMA(2, 0, 0)       0.766938
	*ARIMA(0, 1, 1)       1.100411
	*ARIMA(0, 1, 7)       2.466520
	*ARIMA(3, 1, 2)       2.597065
	*ARIMA(4, 1, 7)       3.115113
	*ARIMA(2, 1, 5)       4.206559
	*ARIMA(6, 1, 7)       8.121447
	*ARIMA(2, 1, 1)      12.550604
	*ARIMA(7, 1, 7)      15.864375
	*ARIMA(2, 1, 4)      27.074888
	*ARIMA(6, 0, 3)      31.510917
	*ARIMA(1, 1, 1)      38.568133
	*ARIMA(0, 1, 2)     191.528097
	*ARIMA(1, 1, 5)    1144.442461
	
- Mean MAPE for different models:
	*ARIMA(4, 1, 7)    0.523198
	*ARIMA(1, 1, 7)    0.543230
	*ARIMA(1, 0, 1)    0.608578
	*ARIMA(1, 0, 5)    0.660343
	*ARIMA(7, 1, 6)    0.778151
	*ARIMA(6, 1, 7)    0.812610
	*ARIMA(1, 1, 5)    0.813446
	*ARIMA(2, 1, 5)    0.936678
	*ARIMA(3, 1, 2)    0.936923
	*ARIMA(0, 1, 7)    0.949785
	*ARIMA(6, 1, 5)    0.994318
	*ARIMA(2, 0, 0)    1.120746
	*ARIMA(1, 1, 1)    1.135333
	*ARIMA(0, 1, 2)    1.166254
	*ARIMA(5, 1, 4)    1.219427
	*ARIMA(2, 1, 4)    1.242152
	*ARIMA(2, 1, 1)    1.404865
	*ARIMA(3, 1, 3)    1.449912
	*ARIMA(7, 1, 7)    1.454671
	*ARIMA(3, 1, 1)    2.251671
	*ARIMA(0, 1, 1)    2.722789
	*ARIMA(6, 0, 3)    3.124057
	
- Number of model occurences:
	*ARIMA(111)    63
	*ARIMA(211)    36
	*ARIMA(012)     6
	*ARIMA(717)     2
	*ARIMA(200)     2
	*ARIMA(311)     2
	*ARIMA(313)     1
	*ARIMA(716)     1
	*ARIMA(617)     1
	*ARIMA(615)     1
	*ARIMA(603)     1
	*ARIMA(514)     1
	*ARIMA(417)     1
	*ARIMA(011)     1
	*ARIMA(312)     1
	*ARIMA(214)     1
	*ARIMA(117)     1
	*ARIMA(115)     1
	*ARIMA(105)     1
	*ARIMA(101)     1
	*ARIMA(017)     1
	*ARIMA(215)     1
	
- MSE by grouped periods:
	fc_period, model_name, MSE	
	7	ARIMA(6, 1, 5)	0.012682
	30	ARIMA(6, 1, 5)	0.018289
	14	ARIMA(3, 1, 1)	0.021743
	30	ARIMA(3, 1, 1)	0.022889
	60	ARIMA(6, 1, 5)	0.023209
	30	ARIMA(1, 1, 7)	0.023445
	60	ARIMA(3, 1, 1)	0.024936
	14	ARIMA(6, 1, 5)	0.027738
	7	ARIMA(1, 0, 1)	0.032471
	14	ARIMA(1, 1, 7)	0.033652
	7	ARIMA(3, 1, 1)	0.037358
	60	ARIMA(1, 1, 7)	0.042279
	7	ARIMA(1, 1, 7)	0.045029
	60	ARIMA(1, 0, 1)	0.069127
	30	ARIMA(1, 0, 1)	0.094577
	14	ARIMA(1, 0, 1)	0.112378
	30	ARIMA(3, 1, 3)	0.143954
	7	ARIMA(7, 1, 6)	0.165500
	60	ARIMA(3, 1, 3)	0.171149
	7	ARIMA(3, 1, 3)	0.186009
	14	ARIMA(7, 1, 6)	0.215247
	30	ARIMA(2, 0, 0)	0.224000
	14	ARIMA(3, 1, 3)	0.234404
	60	ARIMA(2, 0, 0)	0.254374
	14	ARIMA(5, 1, 4)	0.379351
	7	ARIMA(0, 1, 1)	0.414182
	30	ARIMA(7, 1, 6)	0.432477
	7	ARIMA(5, 1, 4)	0.485648
	60	ARIMA(7, 1, 6)	0.523164
	14	ARIMA(1, 0, 5)	0.580493
	60	ARIMA(5, 1, 4)	0.625799
		ARIMA(0, 1, 1)	0.686776
	7	ARIMA(1, 0, 5)	0.733985
	30	ARIMA(5, 1, 4)	0.955112
		ARIMA(0, 1, 1)	1.128721
	7	ARIMA(3, 1, 2)	1.160488
		ARIMA(2, 0, 0)	1.285093
	14	ARIMA(2, 0, 0)	1.304285
	60	ARIMA(3, 1, 2)	1.984334
	14	ARIMA(0, 1, 7)	2.038277
	60	ARIMA(0, 1, 7)	2.094234
	14	ARIMA(0, 1, 1)	2.171967
	60	ARIMA(4, 1, 7)	2.429943
	7	ARIMA(4, 1, 7)	2.544880
	30	ARIMA(3, 1, 2)	2.595818
		ARIMA(0, 1, 7)	2.596050
	7	ARIMA(2, 1, 5)	3.027871
		ARIMA(6, 1, 7)	3.137354
		ARIMA(0, 1, 7)	3.137521
	30	ARIMA(4, 1, 7)	3.390539
	60	ARIMA(2, 1, 5)	4.007218
	14	ARIMA(4, 1, 7)	4.095088
	30	ARIMA(2, 1, 5)	4.380688
	14	ARIMA(3, 1, 2)	4.647619
		ARIMA(2, 1, 5)	5.410461
		ARIMA(6, 1, 7)	5.422032
	60	ARIMA(2, 1, 1)	9.996669
	30	ARIMA(2, 1, 1)	10.012217
		ARIMA(6, 1, 7)	11.257490
	60	ARIMA(7, 1, 7)	12.197356
		ARIMA(6, 1, 7)	12.668911
	14	ARIMA(2, 1, 1)	14.238524
	30	ARIMA(7, 1, 7)	15.632714
	7	ARIMA(2, 1, 1)	15.955005
	14	ARIMA(7, 1, 7)	16.119726
	7	ARIMA(7, 1, 7)	19.507703
	60	ARIMA(2, 1, 4)	20.922005
	14	ARIMA(2, 1, 4)	23.070886
	30	ARIMA(2, 1, 4)	26.581500
	60	ARIMA(1, 1, 1)	28.089860
	7	ARIMA(6, 0, 3)	31.510917
	30	ARIMA(1, 1, 1)	37.496634
	7	ARIMA(2, 1, 4)	37.725163
	14	ARIMA(1, 1, 1)	44.183070
	7	ARIMA(1, 1, 1)	44.502966
	60	ARIMA(0, 1, 2)	128.424736
	30	ARIMA(0, 1, 2)	159.648553
	14	ARIMA(0, 1, 2)	198.660680
	7	ARIMA(0, 1, 2)	279.378419
	60	ARIMA(1, 1, 5)	761.031173
	30	ARIMA(1, 1, 5)	953.838627
	14	ARIMA(1, 1, 5)	1186.922500
	7	ARIMA(1, 1, 5)	1675.977543
	14	ARIMA(6, 0, 3)	NaN
	30	ARIMA(1, 0, 5)	NaN
		ARIMA(6, 0, 3)	NaN
	60	ARIMA(1, 0, 5)	NaN
		ARIMA(6, 0, 3)	NaN

- MAPE by grouped periods:
	fc_period, model_name, MAPE	
	7	ARIMA(1, 0, 1)	0.416239
	30	ARIMA(1, 1, 7)	0.447823
	60	ARIMA(4, 1, 7)	0.457648
	7	ARIMA(6, 1, 7)	0.491324
		ARIMA(4, 1, 7)	0.492433
		ARIMA(7, 1, 6)	0.497791
	60	ARIMA(1, 1, 7)	0.536524
	30	ARIMA(4, 1, 7)	0.546722
	14	ARIMA(1, 1, 7)	0.570732
	60	ARIMA(1, 0, 1)	0.580203
	14	ARIMA(4, 1, 7)	0.595988
	7	ARIMA(1, 1, 7)	0.617841
	60	ARIMA(1, 1, 5)	0.627009
	14	ARIMA(1, 0, 5)	0.631911
	7	ARIMA(2, 1, 5)	0.667623
	14	ARIMA(7, 1, 6)	0.685757
	30	ARIMA(1, 0, 1)	0.687605
	7	ARIMA(1, 0, 5)	0.688775
	14	ARIMA(6, 1, 7)	0.696716
	7	ARIMA(3, 1, 2)	0.705293
	30	ARIMA(1, 1, 5)	0.713793
	7	ARIMA(6, 1, 5)	0.734328
	14	ARIMA(1, 0, 1)	0.750263
		ARIMA(1, 1, 5)	0.826539
	30	ARIMA(7, 1, 6)	0.872531
		ARIMA(3, 1, 2)	0.893471
		ARIMA(6, 1, 5)	0.899694
	60	ARIMA(3, 1, 2)	0.911165
	14	ARIMA(0, 1, 7)	0.921821
	30	ARIMA(0, 1, 7)	0.948559
	60	ARIMA(0, 1, 7)	0.949624
	30	ARIMA(6, 1, 7)	0.978808
	7	ARIMA(0, 1, 7)	0.979135
	30	ARIMA(2, 1, 5)	1.000444
	60	ARIMA(2, 1, 5)	1.001550
	14	ARIMA(2, 0, 0)	1.017727
	60	ARIMA(1, 1, 1)	1.021160
		ARIMA(2, 1, 4)	1.048680
		ARIMA(7, 1, 6)	1.056525
	30	ARIMA(1, 1, 1)	1.059025
	60	ARIMA(6, 1, 5)	1.072403
	14	ARIMA(2, 1, 5)	1.077095
	60	ARIMA(6, 1, 7)	1.083593
	7	ARIMA(1, 1, 5)	1.086443
	60	ARIMA(0, 1, 2)	1.123455
	14	ARIMA(5, 1, 4)	1.124191
	30	ARIMA(2, 0, 0)	1.135226
	60	ARIMA(2, 1, 1)	1.146802
	14	ARIMA(0, 1, 2)	1.155441
	7	ARIMA(2, 0, 0)	1.155540
	30	ARIMA(2, 1, 4)	1.157459
	60	ARIMA(5, 1, 4)	1.169672
	30	ARIMA(0, 1, 2)	1.173859
	60	ARIMA(2, 0, 0)	1.174490
	14	ARIMA(2, 1, 4)	1.188563
		ARIMA(1, 1, 1)	1.198951
	7	ARIMA(5, 1, 4)	1.205758
		ARIMA(0, 1, 2)	1.212261
	14	ARIMA(3, 1, 2)	1.237763
	30	ARIMA(2, 1, 1)	1.249105
	7	ARIMA(1, 1, 1)	1.262196
	14	ARIMA(6, 1, 5)	1.270849
	30	ARIMA(3, 1, 3)	1.298332
	60	ARIMA(7, 1, 7)	1.307000
	7	ARIMA(7, 1, 7)	1.319792
	60	ARIMA(3, 1, 3)	1.329940
	30	ARIMA(5, 1, 4)	1.378087
		ARIMA(7, 1, 7)	1.406981
	7	ARIMA(3, 1, 3)	1.512397
	14	ARIMA(2, 1, 1)	1.551729
	7	ARIMA(2, 1, 4)	1.573909
	14	ARIMA(3, 1, 3)	1.658980
	7	ARIMA(2, 1, 1)	1.671825
	14	ARIMA(7, 1, 7)	1.784913
	7	ARIMA(0, 1, 1)	1.876284
	30	ARIMA(3, 1, 1)	2.053176
	14	ARIMA(3, 1, 1)	2.057130
	60	ARIMA(3, 1, 1)	2.125646
		ARIMA(0, 1, 1)	2.155763
	7	ARIMA(3, 1, 1)	2.770732
	30	ARIMA(0, 1, 1)	2.771920
	7	ARIMA(6, 0, 3)	3.124057
	14	ARIMA(0, 1, 1)	4.087189
		ARIMA(6, 0, 3)	NaN
	30	ARIMA(1, 0, 5)	NaN
		ARIMA(6, 0, 3)	NaN
	60	ARIMA(1, 0, 5)	NaN
		ARIMA(6, 0, 3)	NaN

#### CONCLUSIONS:

- According to MSE and MAPE), best forecasting period is 60 days (the more days, the smaller MSE).
Potwierdza to, że ARIMA lepiej sobie radzi na dłuższych przedziałach czasowych(grubszych częstotliwościach, cite ~notatkiodArka).


- According to MAPE:
	> Best model for period=7: ARIMA(1, 0, 1)
	> Best model for period=14: ARIMA(1, 1, 7)
	> Best model for period=30: ARIMA(4, 1, 7)
	> Best model for period=60: ARIMA(1, 1, 7)

- According to MSE, best ARIMA model is ARIMA(6, 1, 5) (11th by MSE).
- According to MAPE, best ARIMA model is ARIMA(4, 1, 7) (13th by MSE).

- The most important conclusion is number of occurences. ARIMA(1,1,1) occured 63 times, 
ARIMA(2,1,1) 36 times. Third place took ARIMA(0,1,2) - 6 occurences. 
In this case it is better to use MAPE - it will be accurate regardless of model number of occurences. 
- MAPE for ARIMA(1, 1, 1) is 1.135333, 13th in the list. 
- MAPE for ARIMA(2, 1, 1) is 1.404865, 17th in the list. 
- MAPE for ARIMA(0, 1, 2) is 1.166254, 14th in the list. 

Conlusion is simple - we can restrict out p and q ranges into range(0, 3). 
Computations will be much faster, and error will go up just slightly. 



# Experiment_2_architecture_comparison_LSTM (epochs=200)

- LSTM+Dense:
	- epochs=71 (early stopping)
	- MSE_test = 251.18086050927957
	- MAPE_test = 5.793818434456828
	- runtime = 0h 2min 7s

- 2xLSTM+Dense:
	- epochs=200
	- MSE_test = 303.58425092192107
	- MAPE_test = 6.2751841951298015
	- runtime = 0h 14min 51s

- 2xLSTM:
	- epochs=58 (early stopping)
	- MSE_test = 250.15428980532675
	- MAPE_test = 5.838378673515865
	- runtime = 0h 3min 27s

- 3xLSTM:
	- epochs=52 (early stopping)
	- MSE_test = 253.22853446032065
	- MAPE_test = 5.786152079156185
	- runtime = 0h 5min 35s

#### CONCLUSIONS:

- According to MSE, best model is 2xLSTM (3rd by MAPE).
- According to MAPE, best model is 3xLSTM (3rd by MSE).
- According to execution time, the fastest model is LSTM+Dense.
- According to number of epochs, the fastest model to converge is 3xLSTM.

* 2xLSTM+Dense is overfitting! Abandon it! [SHOW PLOT!!]

#### Our main metric is MSE. We will proceed with 2xLSTM model.

# Experiment_3_layers_comparison_RNN_GRU_LSTM(epochs=250)

- 2xGRU:
	- epochs=58 (early stopping)
	- MSE_test = 255.34895213889078
	- MAPE_test = 5.898259475443504
	- runtime = 0h 3min 2s

- 3xGRU:
	- epochs=250
	- MSE_test = 261.32495254787534
	- MAPE_test = 6.414566399239778
	- runtime = 0h 22min 38s

- 2xLSTM:
	- epochs=63 (early stopping)
	- MSE_test = 252.22646472141105
	- MAPE_test = 5.797002623496351
	- runtime = 0h 3min 21s

- 3xLSTM:
	- epochs=54 (early stopping)
	- MSE_test = 255.07601723117122
	- MAPE_test = 5.773877511639204
	- runtime = 0h 4min 57s

#### CONCLUSIONS:

- According to MSE, best model is still 2xLSTM (2rd by MAPE).
- According to MAPE, best model is 3xLSTM (2rd by MSE).
- According to execution time, the fastest model is 2xGRU.
- According to number of epochs, the fastest model is 3xLSTM.

* 3xGRU architecture is overfitting! Abandon it! [SHOW PLOT!!]

#### Our main metric is MSE. We will proceed with 2xLSTM model.

# Experiment_4_testing_hiperparameters_LSTM (architecture: 2xLSTM)

$ optimizer_names = ['RMSprop', 'Adadelta', 'Adagrad', 'Adam', 'SGD']
$ activation_functions = ['tanh', 'sigmoid', 'relu']

4.1 OPTIMIZERS

- RMSprop(): - super zbiega, przy epoce 150 zaczyna się overfitting
	- epochs=250
	- MSE_test = 237.8199280819847
	- MAPE_test = 6.104215985262667
	- runtime = 0h 15min 3s

- Adadelta(): - model się oferfittuje? czy stoi w miejscu?
	- epochs=66 (early stopping)
	- MSE_test = 237.61407042023458
	- MAPE_test = 6.0876205872242215
	- runtime = 0h 3min 39s

- Adagrad(): - model się oferfittuje, definitywnie
	- epochs=49 (early stopping)
	- MSE_test = 239.84077954421502
	- MAPE_test = 6.0563871558483795
	- runtime = 0h 2min 38s

- Adam(): - loss dziwnie...
	* epochs=98 (early stopping)
	* MSE_test = 243.53091955862666
	* MAPE_test = 6.094800264765093
	* runtime = 0h 5min 14s

- SGD(): - loss stoi w miejscu
	- epochs=44 (early stopping)
	- MSE_test = 245.5232741969189
	- MAPE_test = 6.186491760174433
	- runtime = 0h 2min 22s

4.2 ACTIVATION FUNCTIONS

- linear: - stoi w miejscu, nic się nie uczy
	- epochs=27 (early stopping)
	- MSE_test = 1436.8023160811558
	- MAPE_test = 6.13.635869620156674
	- runtime = 0h 1min 38s

- relu: - stoi w miejscu, nic się nie uczy
	- epochs=27 (early stopping)
	- MSE_test = 1513.7802760758168
	- MAPE_test = 13.932933951696572
	- runtime = 0h 1min 31s

- sigmoid: - wykres ładnie zbiega
	- epochs=91 (early stopping)
	- MSE_test = 252.6106345328748
	- MAPE_test = 5.8202346219243575
	- runtime = 0h 5min 8s

- tanh: - wykres ładnie zbiega
	- epochs=75 (early stopping)
	- MSE_test = 249.96305478868283
	- MAPE_test = 5.789534713389976
	- runtime = 0h 4min 21s

#### CONCLUSIONS:

- Adam() optimizer performs the worst.
- According to MSE, best optimizer is Adadelta() (2nd by MAPE).
- According to MAPE, best optimizer is Adagrad() (3rd by MSE).
- According to execution time, the optimizer model is SGD().
- According to number of epochs, the fastest model is also SGD().

- Linear and relu activation do not fit. [SHOW PLOTS! SHOW TABLE!]
- According to MSE and MAPE, best acitvation is tanh. Sigmoid is comparable.
- Execution time and number of epochs are comparable in tanh and sigmoid.

#### Our main metric is MSE. We will proceed with Adadelta() optimizer and tanh activation.

# Experiment_5_input_window_size_LSTM

### Doświadczalne sprawdzenie pokazało, że optimizer Adam() przyspiesza uczenie modelu, osiągając wyniki porównywalne do optimizera Adadelta().

### Dalsze doświadczenia wykonywane są przy użyciu optimizera Adam().

- input_window_size = 60:
	- epochs=85 (early stopping)
	- MSE_test = 205.7574153813731
	- MAPE_test = 5.591417148592397
	- runtime = 0h 3min 20s

- input_window_size = 90:
	- epochs=69 (early stopping)
	- MSE_test = 246.94646587872654
	- MAPE_test = 5.857412664869834
	- runtime = 0h 3min 49s

- input_window_size = 120:
	- epochs=71 (early stopping)
	- MSE_test = 234.1382082989096
	- MAPE_test = 5.313245222856178
	- runtime = 0h 5min 12s

- input_window_size = 150:
	- epochs=76 (early stopping)
	- MSE_test = 181.718114827054
	- MAPE_test = 5.796255001546591
	- runtime = 0h 6min 51s

- input_window_size = 180:
	- epochs=85 (early stopping)
	- MSE_test = 160.24748380680154
	- MAPE_test = 5.4973801788206424
	- runtime = 0h 9min 23s

- input_window_size = 240:
	- epochs=75 (early stopping)
	- MSE_test = 223.12871256965346
	- MAPE_test = 5.364232263430399
	- runtime = 0h 10min 52s

- input_window_size = 360:
	- epochs=58 (early stopping)
	- MSE_test = 257.0968437875267
	- MAPE_test = 5.388198858878602
	- runtime = 0h 12min 59s

- input_window_size = 480:
	- epochs=44 (early stopping)
	- MSE_test = 166.21178481719315
	- MAPE_test = 5.635351572200449
	- runtime = 0h 12min 55s

- input_window_size = 720:
	- epochs=38 (early stopping)
	- MSE_test = 217.99625198526877
	- MAPE_test = 5.637530788122807
	- runtime = 0h 16min 55s

#### CONCLUSIONS:

- Najlepsze MSE dla danych dziennych powstaje dla wejściowego okna czasowego=180 dni.
- Przy ilości dni >180 spada liczba epok oraz wzrasta MSE.
- Powodem jest mniejsza ilość przykładów uczących (im dłuższe okno, tym mniej przykładów uczących).

#### In case of daily data (input_averaging=False),

#### we will proceed with input_window_size=180.


# Experiment_6_input_window_size_with_averaging (1 month = averaged 30 days)
$ input_averaging=True
$ output_averaging=True
$ output_window_size=90

- input_window_size = 12*30 = 360:
	- epochs=68 (early stopping)
	- MSE_test = 649.2439830252079
	- MAPE_test = 11.088632036332589
	- runtime = 0h 0min 33s

- input_window_size = 24*30 = 720:
	- epochs=73 (early stopping)
	- MSE_test = 734.1772747082808
	- MAPE_test = 10.300068137075357
	- runtime = 0h 0min 49s

- input_window_size = 60*30 = 1800:
	- epochs=115 (early stopping)
	- MSE_test = 344.24881129465945
	- MAPE_test = 10.17753705497564
	- runtime = 0h 1min 58s
	
- input_window_size = 90*30 = 2700:
	- epochs=104 (early stopping)
	- MSE_test = 1357.3307074502443
	- MAPE_test = 11.975258503612839
	- runtime = 0h 1min 42s
#### CONCLUSIONS:

- Wykresy loss ładnie zbiegają
- According to MSE (and MAPE), best window size is averaged 60*30 = 1800 days.

#### In case of monthly aggregated data (input_averaging=True, output_averaging=True), 
#### we will proceed with input_window_size=1800. 



# Experiments 7 and 8: Predict 1 day in ARIMA vs LSTM

## Experiment_7_Predict_1_day_ARIMA
$ p = range(0,3)
$ q = range(0,3)
$ series_averaging=False
$ output_window_size = [1]

- MSE_test = 104.31130077559204
- MAPE_test = 1.6366352167704237
- runtime = 0h 11min 04s
- MSE best model: ARIMA(1, 0, 1)
- MAPE best model: ARIMA(2, 0, 1)

Mean MSE for different models
model_name
ARIMA(1, 0, 1)      0.043827
ARIMA(0, 1, 1)      0.138226
ARIMA(2, 0, 1)      0.236413
ARIMA(2, 0, 0)      0.517298
ARIMA(1, 1, 1)     13.301507
ARIMA(2, 1, 2)     34.482887
ARIMA(1, 1, 2)     69.739002
ARIMA(2, 1, 1)    107.741551
ARIMA(0, 1, 2)    712.600997
Name: MSE, dtype: float64
Best model according to MSE:
ARIMA(1, 0, 1) 0.0438274913106985

Mean MAPE for different models
model_name
ARIMA(2, 0, 1)    0.475510
ARIMA(1, 0, 1)    0.685410
ARIMA(1, 1, 1)    1.520423
ARIMA(2, 0, 0)    1.551117
ARIMA(0, 1, 1)    1.855075
ARIMA(2, 1, 1)    1.958379
ARIMA(0, 1, 2)    1.965896
ARIMA(2, 1, 2)    2.210206
ARIMA(1, 1, 2)    2.507700
Name: MAPE, dtype: float64
Best model according to MAPE:
ARIMA(2, 0, 1) 0.4755101869622361

Number of models occurences:
arima
111    63
211    42
012     6
200     2
212     2
011     1
101     1
112     1
201     1
Name: arima, dtype: int64

## Experiment_8_Predict_1_day_LSTM

$ input_averaging=False
$ output_averaging=False
$ output_window_size=180
$ output_window_size=1

- epochs=72 (early stopping)
- MSE_test = 25.564378083241674
- MAPE_test = 1.4566537653226173
- runtime = 0h 7min 22s

#### CONCLUSIONS:
#### In Arima, still best models are ARIMA(1,1,1), ARIMA(2,1,1) and ARIMA(0,1,2).
#### Arima models with p>2 or q>2 were mostly defined as ARIMA(2,1,1). 
#### When predicting 1 day period (daily data), LSTM performs better than ARIMA (both MSE and MAPE).



# Experiments 9 and 10: Predict 30 days in ARIMA vs LSTM
# THIS IS OUT BUSINESS CASE NR 1

## Experiment_9_Predict_30_days_ARIMA

$ p = range(0,3)
$ q = range(0,3)
$ series_averaging=False
$ output_window_size = [30]

- MSE_test = 104.31130077559204
- MAPE_test = 1.6366352167704237
- runtime = 0h 10min 52s
- MSE best model: ARIMA(1, 0, 1)
- MAPE best model: ARIMA(2, 0, 1)

## Experiment_10_Predict_30_days_LSTM
$ input_averaging=False
$ output_averaging=False
$ output_window_size=180
$ output_window_size=30

- epochs=109 (early stopping)
- MSE_test = 160.62472706214174
- MAPE_test = 5.515558528671665
- runtime = 0h 13min 53s

#### CONCLUSIONS:

#### In ARIMA, everything was exactly the same as in Experiment 7 (predicting 1 day).

#### However, LSTM metrics got much worse.

#### When predicting 30 days period (daily data), ARIMA performs better than LSTM. 



# Experiments 11 and 12: Predict 1 month in ARIMA vs LSTM

## Experiment_11_Predict_1_month_ARIMA
> A lot of companies had to be thrown out because the series were too short. 
> Minimum number of months in order to use time series is 30. 

$ p = range(0,3)
$ q = range(0,3)
$ series_averaging=True
$ output_window_size = [1]

- MSE_test = 300.8900315652107
- MAPE_test = 4.28076605394594
- runtime = 0h 00min 31s
- MSE best model: ARIMA(1, 0, 0)
- MAPE best model: ARIMA(1, 0, 0) 

Mean MSE for different models
model_name
ARIMA(1, 0, 0)       0.105779
ARIMA(1, 0, 1)       1.219254
ARIMA(0, 1, 2)       7.052240
ARIMA(1, 1, 2)       7.077079
ARIMA(2, 1, 0)      36.006272
ARIMA(2, 1, 1)     105.933291
ARIMA(2, 1, 2)     201.340467
ARIMA(0, 1, 1)     417.379864
ARIMA(0, 1, 0)     525.648407
ARIMA(1, 1, 0)     876.740593
ARIMA(1, 1, 1)    1131.287100
Name: MSE, dtype: float64
Best model according to MSE:
ARIMA(1, 0, 0) 0.1057792975959577


Mean MAPE for different models
model_name
ARIMA(1, 0, 0)    0.875287
ARIMA(1, 0, 1)    1.095112
ARIMA(1, 1, 2)    2.015496
ARIMA(0, 1, 2)    3.242348
ARIMA(2, 1, 0)    4.063988
ARIMA(0, 1, 0)    4.300492
ARIMA(2, 1, 1)    4.401347
ARIMA(1, 1, 0)    5.564654
ARIMA(1, 1, 1)    6.633860
ARIMA(2, 1, 2)    6.923590
ARIMA(0, 1, 1)    7.972251
Name: MAPE, dtype: float64
Best model according to MAPE:
ARIMA(1, 0, 0) 0.8752870531999042

Number of models occurences:
arima
110    43
010    29
011    20
111     7
210     6
212     4
211     3
100     2
101     2
112     2
012     1
Name: arima, dtype: int64

## Experiment_12_Predict_1_month_LSTM

$ input_averaging=True
$ output_averaging=True
$ output_window_size=1800
$ output_window_size=30

- epochs=128 (early stopping)
- MSE_test = 203.08864091446188
- MAPE_test = 6.897922792776195
- runtime = 0h 2min 17s

#### CONCLUSIONS:

#### ARIMA best model is very simple - it is AR model. Data aggregated to monthly intervals is much easier to predict than daily data. 
#### ARIMA model number of occurences changed. 
#### According to MAE, when predicting 1 month period (monthly data), LSTM performs better than ARIMA.

#### According to MAPE, when predicting 1 month period (monthly data), ARIMA performs better than LSTM. 
#### In case of model comparisons, be base on MAPE as absolute metric. 
#### FINAL CONLUSION: when predicting 1 month period (monthly data), ARIMA performs better than LSTM.



# Experiments 13 and 14: Predict 3 months in ARIMA vs LSTM

# THIS IS OUT BUSINESS CASE NR 2

## Experiment_13_Predict_3_months_ARIMA
$ p = range(0,3)
$ q = range(0,3)
$ series_averaging=True
$ output_window_size = [3]

- MSE_test = 503.32319061497134
- MAPE_test = 5.933649108795311
- runtime = 0h 00min 39s
- MSE best model: ARIMA(1, 0, 0)
- MAPE best model: ARIMA(1, 0, 0) 

Mean MSE for different models
model_name
ARIMA(1, 0, 0)       7.098223
ARIMA(0, 1, 2)      26.188735
ARIMA(2, 1, 0)      39.772127
ARIMA(1, 0, 1)     113.165425
ARIMA(2, 1, 1)     121.006447
ARIMA(0, 1, 1)     188.067810
ARIMA(2, 1, 2)     210.671444
ARIMA(1, 1, 1)     428.844772
ARIMA(1, 1, 0)    1009.827368
ARIMA(0, 1, 0)    1115.180468
ARIMA(1, 1, 2)    2276.732278
Name: MSE, dtype: float64
Best model according to MSE:
ARIMA(1, 0, 0) 7.0982230434727


Mean MAPE for different models
model_name
ARIMA(1, 0, 0)    3.394276
ARIMA(0, 1, 0)    3.827706
ARIMA(2, 1, 1)    4.157569
ARIMA(2, 1, 0)    4.753135
ARIMA(1, 1, 1)    4.774728
ARIMA(1, 1, 0)    4.920343
ARIMA(0, 1, 2)    6.085535
ARIMA(0, 1, 1)    7.191455
ARIMA(1, 0, 1)    8.569313
ARIMA(1, 1, 2)    8.755456
ARIMA(2, 1, 2)    8.840626
Name: MAPE, dtype: float64
Best model according to MAPE:
ARIMA(1, 0, 0) 3.394275892137449

Number of models occurences:
arima
110    43
010    28
011    18
210     6
111     5
212     4
211     3
100     2
101     2
112     2
012     1
Name: arima, dtype: int64

## Experiment_14_Predict_3_months_LSTM
$ input_averaging=True
$ output_averaging=True
$ output_window_size=1800
$ output_window_size=90

- epochs=123 (early stopping)
- MSE_test = 460.83762011980497
- MAPE_test = 10.52658882107607
- runtime = 0h 2min 12s

#### CONCLUSIONS:
#### In ARIMA metrics got higher because of longer period to predict. 
#### Order of models and its occurences are similar as in experiment 11, but a little bit changed. 
#### When predicting 3 months period (monthly data), ARIMA performs better than LSTM (according to MAPE). 



# Experiments 15 and 16: Predict 9 months in ARIMA vs LSTM

# THIS IS OUT BUSINESS CASE NR 3

## Experiment_15_Predict_9_months_ARIMA
$ p = range(0,3)
$ q = range(0,3)
$ series_averaging=True
$ output_window_size = [9]

- MSE_test = 541.9629710600465
- MAPE_test = 7.549712407084164
- runtime = 0h 00min 31s
- MSE best model: ARIMA(1, 0, 0)
- MAPE best model: ARIMA(2, 1, 1)

Mean MSE for different models
model_name
ARIMA(1, 0, 0)      17.428663
ARIMA(2, 1, 0)      43.339895
ARIMA(1, 0, 1)      56.905713
ARIMA(0, 1, 2)      56.968205
ARIMA(2, 1, 1)     119.464692
ARIMA(2, 1, 2)     310.061579
ARIMA(0, 1, 1)     419.523863
ARIMA(1, 1, 1)     626.254950
ARIMA(0, 1, 0)    1223.409356
ARIMA(1, 1, 0)    1396.251884
ARIMA(1, 1, 2)    1691.983882
Name: MSE, dtype: float64
Best model according to MSE:
ARIMA(1, 0, 0) 17.42866286619941

Mean MAPE for different models
model_name
ARIMA(2, 1, 1)     4.071513
ARIMA(0, 1, 0)     4.841777
ARIMA(1, 0, 0)     5.485882
ARIMA(1, 1, 1)     5.794427
ARIMA(2, 1, 0)     5.952930
ARIMA(1, 1, 0)     5.981117
ARIMA(1, 0, 1)     7.961346
ARIMA(0, 1, 2)     9.504315
ARIMA(0, 1, 1)    10.472382
ARIMA(1, 1, 2)    11.005019
ARIMA(2, 1, 2)    11.976130
Name: MAPE, dtype: float64
Best model according to MAPE:
ARIMA(2, 1, 1) 4.071512773221753

Number of models occurences:
arima
110    43
010    28
011    18
210     6
111     5
212     4
211     3
100     2
101     2
112     2
012     1
Name: arima, dtype: int64

## Experiment_16_Predict_9_months_LSTM
$ input_averaging=True
$ output_averaging=True
$ output_window_size=1800
$ output_window_size=270

- epochs=170 (early stopping)
- MSE_test = 2428.150024825217
- MAPE_test = 16.05218845317208
- runtime = 0h 3min 7s

#### CONCLUSIONS:
#### ARIMA metrics once again went higher - because of longer period to predict. 
#### The best model according to MAPE is now ARIMA(2, 1, 1).  
#### Number of occurences is the same as in experiment 13.
#### When predicting 9 months period (monthly data), ARIMA performs better than LSTM (MAPE and MSE). 



# FINAL CONCLUSIONS

## From all experiments!
> In most of the cases, ARIMA performs better than LSTM (MAPE metric). 
> LSTM performed better just in the case of performing 1 day. 
> The longer the output/input period, the worse LSTM results. 
> Basic just on the price feature, ARIMA is better in calculating next predictions as statistical model.
We just need to take in-sample*wyjaśnić period and increase it with prediction after every iteration. 
Thus, every new prediction is based on previous prediction. 
In ARIMA it has no sense to predict >1 period at once. When we tried so, it was straight line. 
So the ARIMA does not predict more days - it duplicates the same value n times. 
That is why we have decided to use loop approach (in order to achieve reasonable predictions). 
> LSTM needs to learn (for some number of epochs), to it would be hard to use similar approach in deep learning as in statistical model. 
In LSTM we predict all the n values at once. That is why LSTM results are worse - it consists of more complicated architecture, which need more features to train. 
Otherwise the LSTM model is in its basic state and is not using its full potential. 
> Learning plots showed, that LSTM is not learning (after 5th epoch). The results is too less number of features (just 1 - price).
[SHOW PLOT when LSTM is not learning!!!]
> Deep learning solution needs more features to train. This may possibly increase model performance. 
It would be worth to check if LSTM model with multiple features can outperform ARIMA model (based just on the price feature). 
> During choosing LSTM features, some of them can be sentiment analysis. 
Such approach may help to achieve interesting model, which will be able to predict sudden spikes in prices and stock market crashes. 

# LAST CONCLUSION:
	- the longer the period, the better ARIMA performs
	- the longer the period, the worse LSTM performs