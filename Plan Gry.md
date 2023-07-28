# problemy Binzesowe

1) predykcja na dniach - 30 dni do przodu
   - Y: 30 kolejnych wartości dniowych
2) predykcja długoterminowa - 3 miesiące do przodu.
   - Y: 3 kolejne wartości miesięczne (uśrednione z dziennych)
3) predykcja długoterminowa #2 - 9 miesiące do przodu.
   - Y: 9 kolejne wartości miesięczne (uśrednione z dziennych)

> 1 miesiąc do przodu jako średnia z mięsiąca - bez sensu (nic nie daje dla giełdera) ~Arek

# Scenariusze do eksperymentów

## Neural Network

### Dane
- input window size
- output window size (zależy od PROBLEMU)
- averaging - zależy od PROBLEMU (np. miesięczne z ARIMY to miesięczne uśrednione dla LSTM)
- price_logarithm = TRUE (ew. [, FALSE])

### Architektura
- LSTM + Dense
- LSTM x2 + Dense
- LSTM x2
- LSTM x3
- GRU + Dense
- GRU x2 + Dense
- GRU x2
- GRU x3

# Hiperparametry
- optimizer
- activation 
- batchsize
- units

## ARIMA

Dane
	miesięcznie albo dziennie (granularność) windows shift
	output window size (PROBLEM)
	averaging ???? (PROBLEM)
	user log ??
Architektura
	p = range(0, 15 + 1)
	q = range(0, 15 + 1)
Hiperparametry
	data_interval = ['daily', 'monthly']
	use_log = yes
	PERIOD = [365]
	P_VALUE_PERCENT = [1%, 5%] - mozna sprawdzic wszystkie 3
	REGRESSION_TYPE_FOR_DF_TEST = {'c'}
	AUTOLAG_FOR_DF_TEST=['AIC', 'BIC'] - mozna sprawdzic
	LJUNGBOX_BOXPIERCE_LAGS=24
	PERIODS_TO_FORECAST = {'daily': [30], 
				'monthly': [3, 6]}


# Analiza Wyników Eksperymentów

Trzeba zrobić MASTER_TABLE 

## Porównywanie Modeli
Możemy porównywać takie wartości jak:
- czas trenowania
- liczba hiperparametrów (do zobaczenia na ARIMie czy jest coś takiego dostępne)
- MSE
- R2 (sprawdzić czy to dobra metryka dla szeregów czasowych)
- Inne metryki dla szeregów czasowych (TODO)
- Wykres funkcji kosztu dla sieci.