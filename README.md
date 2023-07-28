# arima-vs-lstm-on-nasdaq-stock-exchange-data

This study compares the results of two completely different models: statistical one (ARIMA) and deep learning one (LSTM) based on a chosen set of NASDAQ data. Both models are used to predict daily or monthly average prices of chosen companies listed on the NASDAQ stock exchange. Research shows which model performs better in terms of the chosen input data, parameters and number of features. The chosen models were compared using the relative metric mean square error (MSE) and mean absolute percentage error (MAPE). Selected metrics are typically used in regression problems. The performed analysis shows which model achieves better results by comparing the chosen metrics in different models. It is concluded that the ARIMA model performs better than the LSTM model in terms of using just one feature – historical price values – and predicting more than one time period, using the p, q parameters in the range from 0 to 2, Adam optimizer, tanh activation function, and 2xLSTM layer architecture. The longer the data window period, the better ARIMA performs, and the worse LSTM performs. The comparison of the models was made by comparing the values of the MAPE error. When predicting 30 days, ARIMA is about 3.4 times better than LSTM. When predicting an averaged 3 months, ARIMA is about 1.8 times better than LSTM. When predicting an averaged 9 months, ARIMA is about 2.1 times better than LSTM. 
This research was research carried out by D. Kobiela, D. Krefta, W. Król and P. Weichbroth [[1]](#1).

Reasearch poster with the summary of the performed work can be seen in the [research project poster](2022-09-01_AI_TECH_poster.pdf).

![research project poster](https://github.com/DariuszKobiela/arima-vs-lstm-on-nasdaq-stock-exchange-data/2022-09-01_AI_TECH_poster.png)

## References
<a id="1">[1]</a> 
Dariusz Kobiela, Dawid Krefta, Weronika Król, Paweł Weichbroth. 
ARIMA vs LSTM on NASDAQ stock exchange data. 
Procedia Computer Science, Volume 207, 2022, Pages 3836-3845, ISSN 1877-0509. 
https://doi.org/10.1016/j.procs.2022.09.445.
