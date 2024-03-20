import numpy as np

x = np.array([1, 2])
y = np.array([3, 5, 6])

X, Y = np.meshgrid(x, y)
print(X)

percentiles=[.0001, .001, .01]
percentiles+= [1-p for p in percentiles]
print(percentiles)
