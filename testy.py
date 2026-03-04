import numpy as np
import matplotlib.pyplot as plt

dane = np.random.normal(0, 1, 1000)
plt.hist(dane, bins=50,edgecolor = 'black', density=True)
plt.show()



a = np.array([1, 2, np.nan, 4])
b = a[~np.isnan(a)]
print(len(a[~np.isnan(a)]))
print(a)
print(b)
