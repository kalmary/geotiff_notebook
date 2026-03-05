

array = [1, 2, 3, 4, 5]

def funkcja(array):
    big_list = []
    for i, element in enumerate(array):
        new_list = [element for i in range(10000)]
        
        yield new_list

list_gen = funkcja(array)

for element in list_gen:
    a = element
    print(a)
    
