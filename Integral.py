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


# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder, tolerance=1e-20):
    global roots
    if polyorder < 2:
        err = 1  # roots of bad polyorder can not be founded
    else:
        roots = []
        # The polynomials are alternately even and odd functions. So we evaluate only half the number of roots.
        for i in range(1, int(polyorder) // 2 + 1):
            x = np.cos(np.pi * (i - 0.25) / (polyorder + 0.5))
            error = 10 * tolerance
            iters = 0
            while (error > tolerance) and (iters < 1000):
                dx = -Legendre(polyorder, x) / DLegendre(polyorder, x)
                x = x + dx
                iters = iters + 1
                error = abs(dx)
            roots.append(x)

        # Use symmetry to get the other roots
        roots = np.array(roots)
        if polyorder % 2 == 0:
            roots = np.concatenate((-1.0 * roots, roots[::-1]))
        else:
            roots = np.concatenate((-1.0 * roots, [0.0], roots[::-1]))
        err = 0  # successfully roots has been founded
    return [roots, err]


def GaussLegendreWeights(polyorder):
    W = []
    [xis, err] = LegendreRoots(polyorder)
    if err == 0:
        W = 2.0 / ((1.0 - xis ** 2) * (DLegendre(polyorder, xis) ** 2))
        err = 0
    else:
        err = 1  # we couldnt find any roots then we have no weights
    return [W, xis, err]