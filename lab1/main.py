import numpy as np

A = np.loadtxt("../Matrix/matrixA.txt")
B = np.loadtxt("../Matrix/matrixB.txt")
C_computed = np.loadtxt("../Matrix/resultMatrix.txt")

C_expected = A @ B

if np.allclose(C_expected, C_computed):
    print("Ок")
else:
    print("Не ок")
