# ARIMA PROCESS AUTOMATION

# STAŁE:
- poziom istotności p-value, na np. 5%

# 1. Wczytują kolejną spółkę z pętli. Zapisuję wykres szeregu. Wyświetlam dekompozycję szeregu przed zlogarytmowaniem. 
- DEPRECATED: Użytkownik wybiera czy logarytmować, czy nie. Jesli był logarytm, to zapisuję to do zmiennej
- użytkownik wybiera rodzaj wykresu (c lub ct lub ctt lub nc). Do wyboru: 
    - ***
    regression : {'c','ct','ctt','nc'}
    Constant and trend order to include in regression.

    * 'c' : constant only (default).
    * 'ct' : constant and trend.
    * 'ctt' : constant, and linear and quadratic trend.
    * 'nc' : no constant, no trend.
    - ***
    autolag : {'AIC', 'BIC', 't-stat', None}
    Method to use when automatically determining the lag.

    * if None, then maxlag lags are used.
    * if 'AIC' (default) or 'BIC', then the number of lags is chosen
      to minimize the corresponding information criterion.
    * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
      lag until the t-statistic on the last lag length is significant
      using a 5%-sized test.    
    - ***
- UZYĆ NIEFORMALNE ZAŁOŻENIE: logarytmujemy wszystkie spółki. Bo: 
    - I tak większość trzeba by zlogarytmować
    - Jeśli nie trzeba jakiejś spółki logarytmować, to logarytm nic nie psuje, nie zmienia
    - przyspiesza to proces automatyzacji
	
# 2. Test Dickeya-Fullera na szeregu z punktu 1

# 3. 
- Jeśli w kroku 2 wyszła niestacjonarność, to różnicowanie jednokrotne. Sprawdź, czy różnicowanie przyniosło efekt poprzez ponowne wykonanie testu DF. 
    - Jesli przyniosło efekt: idź dalej do punktu 4. 
    - Jeśli nie przyniosło efektu: to kolejne różnicowane. Znowu sprawdzamy przez wykonanie testu. Pętla. 
- Jeśli wyszła stacjonarność, to pomiń. 

# 4. Przeprowadzenie automatycznego wyboru parametrów p, d, q. Wybór kryterium: AIC, oraz powtórzyć dla BIC
- https://www.kaggle.com/nholloway/deconstructing-arima
- parametr d automatycznie ustalony na podstawie ilości różnicowań
- p = range(0, 10)
- q = range(0, 10)

# 5. Automatyczne sprawdzenie białego szumu. 
- Jeśli lb>p-value and bx>p-value, to idź dalej
- Jeśli lb<p-value or bx<p-value (czyli reszty nie są białym szumem), to:
    - Wyświetl wykresy ACF i PACF. Użytkownik ocenia, czy reszty są białym szumem. 
    - Jeśli użytkownik zaakceputuje, to idź dalej
    - Jeśli użytkownik nie zaakceptuje, to PORZUĆ MODEL!!! (BREAK THE PROCESS, + INFO: NIE MOŻNA DOKONAĆ PROGNOZY)

# 6.  Przeprowadzenie automatycznej prognozy ARIMA: 
- okres prognozy: zgodny z siecią neuronową. Przyjmijmy okres stały n=3. Obok w tabelcę zapisuję max{p,q} jako sugerowany okres. 
- jesli AIC i BIC wskazują na tą samą ARIMĘ, to wykonanie jednej prognozy
- jeśli AIC i BIC wskazują na różne ARIMY, to wykonanie dwóch prognoz
- wybranie lepszego jakościowo wyniku

# 7. Zapisanie wyników:
- przed wygenerowaniem wykresów, dokonać operacji odwrotnej do logarytmowania, czyli exp() - exponent (wykładnik potęgowy), 
- wygenerowanie i zapisanie wykresów (ogólny i szczegółowy, tylko ostatnie okresy prognozy)
- statystyki jakości prognozy (mse), druga (np. mae lub rmse) tylko jeśli w sieci neurnowej użyłem dwóch
- tabelka z true value i predicted value
- DEPRECATED: ZAPISUJĘ LISTĘ SPÓŁEK, które były a które nie zlogarytmowe