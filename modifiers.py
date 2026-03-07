# 3. stworz folderze projektu plik Modifiers.py:
# 3.1 Stwórz w nim klasę NDVIdecrease - niech (póki co) argumentem w metodzie __init__ będzie ndvi_array (typ: np.ndarray)
# 3.2 Klasa ma posiadać metody, które zmodyfikują ndvi_array. Póki co nie będziemy tych metod rozpisywać dokładnie, ale:
# 3.2.1 Przygotuj kilka metod (same nazwy i opis metod, co robią, w komentarzach pod nimi), które będą modyfikować (symulować zniszczenia pól) macierz ndvi.
#       Żeby program nie zgłaszał błędu, każda metoda powinna printować rozmiar ndvi_array.
# 3.3 Zaimportuj klasę do main i po wczytaniu pliku tif wg. punktu 1., użyj każdej metody. 
import numpy as np

# wartosc dodana pracy - nie samo okreslenie spadku wskaznika ndvi ale na tej podstawie szacunek np. procentowego zniszczenia pola
# a co za tym idzie mniejszymi plonami?

class NDVIdecrease:
    def __init__ (self, ndvi_array: np.ndarray): 
        self.ndvi_array = ndvi_array

    def boars (self):
        print(self.ndvi_array)
    # dziki ryjace w polu i niszczace je, mysle, ze na zasadzie pedzla (wielkosc do ustalenia) ktory znaczaco obniza NDVI
    # w danym miejscu + pewnie jakas sciezka od boku pola. do pomyslenia czy jakos symulujemy zachowania stada czy pojedyncze dziki

    def storm (self):
        print(self.ndvi_array)
    # burza niszczaca pole poprzez mocne wiatry (?)
    # znisczenie poprzez rozmyty duzy pedzel chyba
    
    def drought (self):
        print(self.ndvi_array)
    # susza suszy 
    # obnizenie calego indeksu na polu, ale moze gradientowo tak, w czescie polnocnej bardziej niz 
