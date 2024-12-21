"""
Sensor functions defined in [Olivier 2011].

Olivier, Géraldine. Anisotropic metric-based mesh
adaptation for unsteady CFD simulations involving
moving geometries. Diss. 2011.
"""

import ufl

__all__ = ["bowl", "hyperbolic", "multiscale", "interweaved"]


def bowl(*coords):
    return 0.5 * sum([xi**2 for xi in coords])


def hyperbolic(x, y):
    return ufl.conditional(
        abs(x * y) < 2 * ufl.pi / 50, 0.01 * ufl.sin(50 * x * y), ufl.sin(50 * x * y)
    )


def multiscale(x, y):
    return 0.1 * ufl.sin(50 * x) + ufl.atan(0.1 / (ufl.sin(5 * y) - 2 * x))


def interweaved(x, y):
    return ufl.atan(0.1 / (ufl.sin(5 * y) - 2 * x)) + ufl.atan(
        0.5 / (ufl.sin(3 * y) - 7 * x)
    )
