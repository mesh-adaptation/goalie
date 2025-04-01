"""
Problem specification for a simple steady state flow-past-a-cylinder test case which
solves a Stokes problem.

The test case is notable for Goalie because the prognostic equation is nonlinear.

Code here is based on that found at
    https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/06-pde-constrained-optimisation.ipynb
"""

import os

import ufl
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.mesh import Mesh
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunctions

from goalie.field import Field

mesh = Mesh(os.path.join(os.path.dirname(__file__), "mesh-with-hole.msh"))
fields = [Field("up", unsteady=False)]
dt = 1.0
end_time = dt
dt_per_export = 1
num_subintervals = 1
steady = True


def get_function_spaces(mesh):
    r"""Taylor-Hood :math:`\mathbb P2-\mathbb P1` space."""
    return {"up": VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)}


def get_solver(self):
    """Stokes problem solved using a direct method."""

    def solver(i):
        W = self.function_spaces["up"][i]
        up = self.field_functions["up"]

        # Define variational problem
        R = FunctionSpace(self[i], "R", 0)
        nu = Function(R).assign(1.0)
        u, p = ufl.split(up)
        v, q = TestFunctions(W)
        F = (
            ufl.inner(ufl.dot(u, ufl.nabla_grad(u)), v) * ufl.dx
            + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            - ufl.inner(q, ufl.div(u)) * ufl.dx
        )

        # Define inflow and no-slip boundary conditions
        y = ufl.SpatialCoordinate(self[i])[1]
        u_inflow = ufl.as_vector([y * (10 - y) / 25.0, 0])
        noslip = DirichletBC(W.sub(0), (0, 0), (3, 5))
        inflow = DirichletBC(W.sub(0), Function(W.sub(0)).interpolate(u_inflow), 1)
        bcs = [inflow, noslip, DirichletBC(W.sub(0), 0, 4)]

        # Solve
        sp = {
            "mat_type": "aij",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_shift_type": "inblocks",
        }
        solve(
            F == 0,
            up,
            bcs=bcs,
            solver_parameters=sp,
            ad_block_tag="up",
        )
        yield

    return solver


def get_initial_condition(self):
    """
    Dummy initial condition function which acts merely to pass over the
    :class:`FunctionSpace`.
    """
    x, y = ufl.SpatialCoordinate(self[0])
    u_inflow = ufl.as_vector([y * (10 - y) / 25.0, 0])
    up = Function(self.function_spaces["up"][0])
    u, p = up.subfunctions
    u.interpolate(u_inflow)
    return {"up": up}


def get_qoi(self, i):
    """Quantity of interest which integrates pressure over the boundary of the hole."""

    def steady_qoi():
        u, p = ufl.split(self.field_functions["up"])
        return p * ufl.ds(4)

    return steady_qoi
