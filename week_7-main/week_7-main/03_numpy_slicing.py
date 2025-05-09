import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print("Original Array:\n", a)
print("Slicing from 2nd row:\n", a[1:])
print("Second column:\n", a[:, 1])
print("Second row:\n", a[1, :])
print("Column 1 onwards:\n", a[:, 1:])