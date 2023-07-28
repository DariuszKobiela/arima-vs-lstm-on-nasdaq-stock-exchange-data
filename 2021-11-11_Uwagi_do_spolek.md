### Programy R i python wykrywają sezonowość w każdej spółce. 
Uzasadnienie:
Zakładamy, że wszystkie spółki giełdowe nie mają sezonowości.
Mamy natomiast Random Walk (błądzenie przypadkowe) - losowe skoki tych akcji. 
Programy jako "sezonowość" wykrywają właśnie ten Random Walk?
Najlepiej usunąć wykres sezonowości z dekompozycji w pythonie. 
Nikt się nie przyczepi, bo z założenia w spółkach jest brak sezonowości (co potwierdza program JDemetra+). 

## Program JDemetra+ jako szanowane narzędzie statystyczne. 
Dać wykres sezonowości (linia prosta). 
Linia prosta wskazuje na brak sezonowości. 

## Coś złego stało się z danymi NASDAQ w 2015 roku. 
Wiekszość wadliwych spółek ma błąd od (lub w tym) właśnie roku. 
Spółki posiadające ten błąd mają też błędne osie (zła wartość ceny [\$]).

## Metryka do ARIMY - użyć MAPE (uwzględnia różnice w skalach spółek)
Inaczej MSE musi być liczone z uwzględnieniem wag. 
https://web.sgh.waw.pl/~atoroj/ekonometria/cwiczenia_04.pdf

## Zapisywać dane!!
potem pd.to_latex() - generowanie tabelek
I można poprawić złe wykresy, potem zrobić wszystko z danymi. 
