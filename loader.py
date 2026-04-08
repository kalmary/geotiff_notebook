# 1. utwórz w folderze projektu nowy plik: Loader.py
# 1.1 Niech plik zawiera funkcję load_data. 
# 1.2 Funkcja musi przyjmować (wskaż typ) argument source_dir (typ string) i extension (typ string). Musi przetwarzać tego stringa do pth.Path.
#     Następnie niech stworzy listę ścieżek do wszystkich plików z wybramym rozszerzeniem, znajdujących się w tym folderze.
# 1.3 W forze wczytaj pliki z użyciem rasterio
# 1.4 Po wczytaniu pliku funkcja ma go zwrócić (cały dataset) i dopiero wtedy wczytywać kolejny plik.

from pathlib import Path
import rasterio as rio

def load_data (source_dir: str, extension: str):
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Folder {source_dir} nie istnieje")
    
    pattern = f"*{extension}"

    path_list = list(source_dir.glob(pattern))

    # print(path)
    # print(path.resolve())
    # print(list(path.iterdir()))
    for path in path_list:
        dataset = rio.open(path)
        yield dataset, path
       
