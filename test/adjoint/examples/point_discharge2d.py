"""
Problem specification for a simple advection-diffusion test case with a point source,
from [Riadh et al. 2014].

This test case is notable for Goalie because it has an analytical solution, meaning the
effectivity index can be computed.

[Riadh et al. 2014] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system: 2D
    hydrodynamics TELEMAC-2D software release 7.0 user manual." Paris: R&D, Electricite
    de France, p. 134 (2014).
"""

import numpy as np
import ufl
from finat.ufl import FiniteElement
from firedrake.assemble import assemble
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.solving import solve
from firedrake.ufl_expr import CellSize, TestFunction
from firedrake.utility_meshes import RectangleMesh

from goalie.field import Field
from goalie.math import bessk0

# Problem setup
n = 0
mesh = RectangleMesh(100 * 2**n, 20 * 2**n, 50, 10)
finite_element = FiniteElement("Lagrange", ufl.triangle, 1)
fields = [Field("tracer_2d", finite_element=finite_element, unsteady=False)]
end_time = 1.0
dt = 1.0
dt_per_export = 1
src_x, src_y, src_r = 2.0, 5.0, 0.05606388
rec_x, rec_y, rec_r = 20.0, 7.5, 0.5
steady = True
get_initial_condition = None


def source(mesh):
    """
    Gaussian approximation to a point source at (2, 5) with discharge rate 100 on a
    given mesh.
    """
    x, y = ufl.SpatialCoordinate(mesh)
    return 100.0 * ufl.exp(-((x - src_x) ** 2 + (y - src_y) ** 2) / src_r**2)


def get_solver(self):
    """Advection-diffusion equation solved using a direct method."""

    def solver(i):
        fs = self.function_spaces["tracer_2d"][i]
        c = self.field_functions["tracer_2d"]

        # Define constants
        fs = self.function_spaces["tracer_2d"][i]
        R = FunctionSpace(self[i], "R", 0)
        D = Function(R).assign(0.1)
        u_x = Function(R).assign(1.0)
        u_y = Function(R).assign(0.0)
        u = ufl.as_vector([u_x, u_y])
        h = CellSize(self[i])
        S = source(self[i])

        # Stabilisation parameter
        unorm = ufl.sqrt(ufl.dot(u, u))
        tau = 0.5 * h / unorm
        tau = ufl.min_value(tau, unorm * h / (6 * D))

        # Setup variational problem
        psi = TestFunction(fs)
        psi = psi + tau * ufl.dot(u, ufl.grad(psi))
        F = (
            S * psi * ufl.dx
            - ufl.dot(u, ufl.grad(c)) * psi * ufl.dx
            - ufl.inner(D * ufl.grad(c), ufl.grad(psi)) * ufl.dx
        )

        # Zero Dirichlet condition on the left-hand (inlet) boundary
        bc = DirichletBC(fs, 0, 1)

        # Solve
        sp = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        solve(F == 0, c, bcs=bc, solver_parameters=sp, ad_block_tag="tracer_2d")
        yield

    return solver


def get_qoi(self, i):
    """
    Quantity of interest which integrates the tracer concentration over an offset
    receiver region.
    """

    def steady_qoi():
        c = self.field_functions["tracer_2d"]
        x, y = ufl.SpatialCoordinate(self[i])
        kernel = ufl.conditional((x - rec_x) ** 2 + (y - rec_y) ** 2 < rec_r**2, 1, 0)
        area = assemble(kernel * ufl.dx)
        area_analytical = ufl.pi * rec_r**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_analytical / area
        return scaling * kernel * c * ufl.dx

    return steady_qoi


def analytical_solution(mesh):
    """
    Analytical solution as represented on a given mesh. See [Riadh et al. 2014].
    """
    x, y = ufl.SpatialCoordinate(mesh)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(R).assign(1.0)
    D = Function(R).assign(0.1)
    Pe = 0.5 * u / D
    r = ufl.max_value(ufl.sqrt((x - src_x) ** 2 + (y - src_y) ** 2), src_r)
    return 0.5 / (ufl.pi * D) * ufl.exp(Pe * (x - src_x)) * bessk0(Pe * r)
