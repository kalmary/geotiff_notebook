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

import rasterio as rio
import matplotlib 
from matplotlib import pyplot as plt
import numpy as np 
import pathlib as pb
from pathlib import Path


# 1
with rio.open('data/czarna 121 2025-04-22-ORTHO-NDVI.data.tif') as dataset:
    array = dataset.read(1)
    nodata = dataset.nodata

# mozna tez tak
dataset = rio.open('data/czarna 121 2025-04-22-ORTHO-NDVI.data.tif')


    # pozdrawiam
    
# print (dataset.name)
# print (dataset.count)
# print (dataset.width)
# print (dataset.height)
# print (array.shape)


# 2

# print(array)
array = np.where(array == nodata, np.nan, array) 

plt.imshow(array, 
              cmap='RdYlGn', 
              vmin= -1,
              vmax = 1)
plt.savefig('ndvi.png', dpi=300) # zapisuje obrazek do pliku, zeby nie tracic jakosci
plt.show()

# 3 'nodata': - 10000

meta = dataset.meta

print("metadane\n\n", meta)
print("bounding box\n\n", dataset.bounds)

# 4

print("wielkosc macierzy\n\n", array.shape)
print("min, max, mean wartosc macierzy\n", np.nanmin(array), np.nanmax(array), np.nanmean(array))

#5

# def zabawa5(datafile , extension):
#         idk = []
#         p = Path(datafile)
#         for x in p.iterdir():
#             if x.suffix == extension:
#                 idk.append(x)

#         return(idk)



# lista = zabawa5('geotiff_notebook/data', '.tif')

path2file = Path('data')
ext = '.tif'
path_list = list(path2file.glob(f'*{ext}'))
print(path_list)
# test
# test
# test


# def zabawa52(do poprawy):
#       
#        for x in p.glob("*.tif"):
#                idk2.append(x)
#
#        return(idk2)

# zabawa52()
# print(idk2)

# 6  z density=True wychodza mi wartosci prawdopodobienstwa ponad 1 (chyba bo bardzo ciasne bins, wiec bez density jest)
array_clean = array[~np.isnan(array)]
plt.figure(figsize=(8,5))

plt.title('umiem dodawac tytul jej')
plt.xlabel('wartosci indeksu NDVI')
plt.ylabel('liczba razy wystapienia danej wartosci')
plt.hist(array_clean, bins=100, edgecolor = 'black')
plt.show() 

