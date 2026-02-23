# przerzuć wszystkie dane do /data. Zwróć uwagę, że gitignore sprawia, że nie wrzucisz ich do repozytorium (to dobrze)
# stwórz nowe wirtualne środowisko i aktywuj je
# zainstaluj potrzebne biblioteki - zwróć uwagę, że masz pliik requirements.txt -> żeby go użyć wklej w terminal: pip install -r requirements.txt

# poczytaj o:
# 1 - numpy
# 2 - matplotlib
# 3 - rasterio - niewiele, używamy jej w konkretny i ograniczony sposób
# 4 - scikit-learn (bardzo mało, to zaawanowana biblioteka, ale póóóóźniej jej użyjemy)

# spróbuj zrobić kod z użyciem numpy, matplotlib i rasterio, który:
# 1 - wczyta dane z pliku .tif (użyj rasterio)
# 2 - wyświetli dane jako obraz (użyj matplotlib)
# 3 - wyświetli metadane (użyj rasterio)
# 4 - wyprintuj jaki jest rozmiar macierzy, którą chcesz wyświetlić, oraz jakie są minimalne, maksymalne i średnie wartości pikseli (użyj numpy)
# 5 - napisz oddzielną funkcję, która (zadanie z gwiazdką XD):
## 5.1 - przyjmie argumenty: ścieżkę do folderu z danymi /data (typ: string lub pathlib.Path) oraz rozszerzenie pliku np. ".tif" (string)
## 5.2 - zwróci listę ścieżek do wszystkich plików z danym rozszerzeniem w podanym folderze (użyj pathlib)
# 6 - wyświetli histogram wartości pikseli (użyj matplotlib) * zadanie z gwiazdką
