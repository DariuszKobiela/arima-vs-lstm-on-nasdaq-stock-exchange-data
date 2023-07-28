EXPERIMENT 2:
- Jeżeli przewidujemy sekwencje, to lepiej sobie radzą sieci, które w ostatniej warstwie maja wartwę rekurencyną (bo ona zwraca sekwencję ,a nie wektor jak warstwa gęsta Sense). 
W seknwncjach jest przechowyana informacja o zależnościach między kolejnymi puntami  tej sekwencji. 
W warstwie Dense nie istnieją żadne zaleznosći tego typu. 


EXPERIMENT 3:
3xGru MAPE - źle zregularyzowana (tak duża sieć). Domyslne wartości elementów regularyzujących, np. Adam() parameters były niewłaściwe dla tej sieci. 
Dlatego zbieganie nie było monotoniczne, model wariował. Był zbyt wrazliwy na działąnie gradientów. Dlatego tak drstycznie zmieniały się wartości funkcji kosztu. Powodowało to, że sieć się nei uczyła.  
Można by spróbować powtórzyć eksperyment z innymi wartościami optimizera (beta=?, learning rate=?). 

3xGRU MSE - przykład overfittingu

MAPE 3xLSTM - wywal ogon z wykresu, zostaw do 20 epoki. Powiedz, że potem nie było żadnej znaczącej poprawy (Early Stopping daje +25 epok). 
Stwórz na nowo te wykresy!!

EXPERIMENT 4:
Super obrazek do pracy (ale jest on czystym przypadkiem):
-> plot PPC company id_1 tanh_Adam_2021_11-26_00_22
Daj to do pracy. 

EXPERIMENT 5:
-> 180 okno

EXPERIMENT 6:
-> Wykresy w końcu wyglądają jak powinny, ale tego NIE PISZ W PRACY (bo wtedy sobie rozwalisz inne wykresy, wystawisz się na krytykę recenzenta).
BĄDZ OSTROŻNY, CO PISZESZ. 


EXPERIMENT 7 i 8:
-> Dla 30 dni LSTM potrzebuje więcej danych, więc radzi sobie gorzej niż ARIMA. 
-> Dla 1 dnia (problem regresji) ilość danych była wystarczjąca dla LSTM do osiągnięcia lepszego performance niż ARIMA. 
-> Sieć sobie radzi lepiej niż ARIMA przy predykcji 1 dnia, gdyż wtedy mogą wykorzystać swój potencjał. 


EXPERIMENT 9 i 10:
-> Przy przewidywaniu sekwencji ilość danych była niewystarczająca dla LSTM.  


EXPERIMENT 11 i 12:
-> coś jak regresja
-> jest to idealny case dla ARIMY (dane miesięczne). ARIMA lepiej sobie radzi na okresach długoterminowych (miesiąc, kwartał, rok). 
Cytat artykolow naukowych! Potwierdzamy wnioski z papera!!! 

EXPERIMENT 13 i 14:
-> Podobnyk experymetn i podobne wyniki jak w EXP 11 i 12. Znowu potwierdza ten paper. 
-> PISZ EWENTUALNIE: ARIMA dobrze trafia w średnie punkty, a niedotrenowany LSTM radzi sobie gorzej. 
-> LSTM skuteczniej wyłapuje anomalie, ale średnio gorzej sobie radzi. ARIMA średnio lepiej sobie radzi, ale słabo wykrywa anomalie. 
-> Dlatego mse jest czasawi wyższe, bo mse bardzo każde pojedyncze rozbieżności.
-> LSTM można wykorzystać do wyłapywania anomalii.  
-> Tutaj DODATKOWO MSE wykres (bo on dał inne wyniki niż MAPE, bo bardziej każe rozbieżności). 

EXPERIMENT 15 i 16:
-> Znowu potwierdza wyniki z EXP 11,12 i EXP 13,14. 
-> ARIMA dużo lepiej (dalekie przeiwydanie do przodu). 


FINAL CONCLUSIONS:
- wykres może być czytelny, a nie ładny
- Dlaczego się zdecydowałem na dane rozwiązanie - pisz w sekcji "Założenia projektowe" (wprowadzenie do eksperymentu). 
- ciężko oplem corsą zaorać pole. Porównujesz traktor i opel corsa w innych kategoriach
- STARAJ SIĘBYĆ PRECYZYJNY - NIE OGÓLNY, BO ŁATWO SIĘ PRZYCZEPIĆ!!!
- RÓB WIELE WYKRESÓW


DLA KAŻDEGO EXPERYMENTU:
- arima kolor zielony, lstm czerwony (te same przez całą pracę)
- wykres bar chart: porównanie MAPE dla obu modeli (LSTM i ARIMA). ROB WYKRESY - dobrze dla oczu!!

EXPERYMENT TESTOWANIE HIPERPARAMETROW i testowanie LSTM-GRU:
- zrobić wykres val_loss (mse): na jednym wykresie porównaj val_mse dla LSTM i GRU
- na drugim wykresie porówanj val_mse dla 4 różnych parametrów

OSTATNI RODZIAŁ:
-> Być może sieć poradziła by sobie lepiej bez logarytmu. Można by też sprawdzić inny sposób próbkowania danych. 

DODATKOWO:
- wyjaśnić pojęcie in sample (w tekście). 