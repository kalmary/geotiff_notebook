# 2. W pliku main.py:
# 2.1 Stwórz strukturę która uniknie zmiennych i funkcji globalnych (def main + if __name__ == "__main__": main())
# 2.2 Zaimportuj load data
# 2.3 W def main() wywołaj funkcję load_data - funkcja musi działać w pętli. Splotuj ndvi (tak jak ostatnio) kolejno dla każdego kolejnego wykresu. 
#     Z gwiazdką: wykresy muszą pokazać się wszystkie naraz (który element odpowiada za ich wyświetlanie?)

import loader
from loader import load_data
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt 

def main():

    # print(type(load_data))
    # gen = load_data('data', '.tif')
    # print(gen)
    for dataset in load_data('data', '.tif'):
        array = dataset.read(1)
        nodata = dataset.nodata
        array = np.where(array == nodata, np.nan, array) 
        plt.figure(figsize=(8,5)) # adding this makes all the figures appear in separate windows, idk why but seems to be working xd
        plt.imshow(array, 
              cmap='RdYlGn', 
              vmin= -1,
              vmax = 1)
        # plt.savefig('ndvi.png', dpi=300) # zapisuje obrazek do pliku, zeby nie tracic jakosci
        
        print (dataset)

    plt.show()

if __name__ == "__main__":
    main()

    
