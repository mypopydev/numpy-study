import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("Creation from array:\n", A)
print("Transpose A:\n", A.T)
print("Det A:\n", np.linalg.det(A))

try:
    inv_A = np.linalg.inv(A)
    print("Inverse A:\n", inv_A)
    print("Check Inverse:\n", A @ inv_A)
except np.linalg.LinAlgError:
    print("Matrix A is singular and cannot be inverted.")

B = np.arange(9).reshape(3, 3)
print("Creation from arange:\n", B)
