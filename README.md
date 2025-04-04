# Algorytm rozpoznawający płeć

Opis działania algorytmu służącego do rozpoznawania płci z pliku głosowego:
Algorytm rozpoznawania płci z mowy analizuje sygnał dźwiękowy, wykorzystując przybliżenie
podstawowej częstotliwość głosu (f0) oraz pierwszy formant (F1). Po wczytaniu pliku WAV sygnał jest
redukowany do mono (jeśli to konieczne), normalizowany i przetwarzany z użyciem okna Hanninga w
celu redukcji efektów przecieków widma. Następnie, za pomocą szybkiej transformaty Fouriera (FFT),
wyznaczane jest widmo amplitudowe, a odwrotna transformata Fouriera (IFFT) pozwala obliczyć
cepstrum, które służy do identyfikacji dominującej częstotliwości w zakresie f0 (80–250 Hz).
Przybliżenie f0 jest wyznaczane jako średnia odwrotności 15 dominujących quefrencji w cepstrum, co
uwzględnia sąsiednie składowe harmoniczne. Średnia ta pozwala uśrednić wpływ drobnych
odchyleń, które mogą wynikać z szumu lub usterek w nagraniach. Dodatkowo algorytm estymuje
pierwszy formant F1, identyfikując szczyt w widmie amplitudowym w zakresie 300–3500 Hz. Na
podstawie wartości przybliżenia f0 i F1 algorytm klasyfikuje płeć: głosy o niskim (f0 (80–160 Hz) and
(F1 poniżej 800 Hz)) klasyfikuje jako męskie, a głosy o wyższym f0 (185–250 Hz) jako kobiece, resztę
(160 -185 Hz) klasyfikuje na podstawie F1, gdy jest mniejsze od 500 Hz głos jest uznawany jako męski,
w przeciwnym razie kobiecy.
Jak widać poniżej algorytm zadziałał z trafnością klasyfikowania wynoszącą około 87%