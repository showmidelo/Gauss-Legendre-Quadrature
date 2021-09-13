import numpy as np
# calculate Gauss Legendre Quadrature


# Recursive generation of the Legendre polynomial of order n
def Legendre(n, x):
    x = np.array(x)
    if n == 0:
        return x * 0 + 1.0
    elif n == 1:
        return x
    else:
        return ((2.0 * n - 1.0) * x * Legendre(n - 1, x) - (n - 1) * Legendre(n - 2, x)) / n


# Derivative of the Legendre polynomials
def DLegendre(n, x):
    x = np.array(x)
    if n == 0:
        return x * 0
    elif n == 1:
        return x * 0 + 1.0
    else:
        return (n / (x ** 2 - 1.0)) * (x * Legendre(n, x) - Legendre(n - 1, x))