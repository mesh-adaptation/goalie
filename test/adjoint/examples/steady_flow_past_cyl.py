"""
Problem specification for a simple steady
state flow-past-a-cylinder test case which
solves a Stokes problem.

The test case is notable for Goalie
because the prognostic equation is
nonlinear.

Code here is based on that found at
    https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/06-pde-constrained-optimisation.ipynb
"""

import os

from firedrake import *
from firedrake.__future__ import interpolate

mesh = Mesh(os.path.join(os.path.dirname(__file__), "mesh-with-hole.msh"))
fields = ["up"]
dt = 1.0
end_time = dt
dt_per_export = 1
num_subintervals = 1
steady = True


def get_function_spaces(mesh):
    r"""
    Taylor-Hood :math:`\mathbb P2-\mathbb P1` space.
    """
    return {"up": VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)}


def get_solver(self):
    """
    Stokes problem solved using a
    direct method.
    """

    def solver(i):
        W = self.function_spaces["up"][i]
        up = self.fields["up"]

        # Define variational problem
        R = FunctionSpace(self[i], "R", 0)
        nu = Function(R).assign(1.0)
        u, p = split(up)
        v, q = TestFunctions(W)
        F = (
            inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
            - inner(p, div(v)) * dx
            - inner(q, div(u)) * dx
        )

        # Define inflow and no-slip boundary conditions
        y = SpatialCoordinate(self[i])[1]
        u_inflow = as_vector([y * (10 - y) / 25.0, 0])
        noslip = DirichletBC(W.sub(0), (0, 0), (3, 5))
        inflow = DirichletBC(W.sub(0), assemble(interpolate(u_inflow, W.sub(0))), 1)
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
    Dummy initial condition function which
    acts merely to pass over the
    :class:`FunctionSpace`.
    """
    x, y = SpatialCoordinate(self[0])
    u_inflow = as_vector([y * (10 - y) / 25.0, 0])
    up = Function(self.function_spaces["up"][0])
    u, p = up.subfunctions
    u.interpolate(u_inflow)
    return {"up": up}


def get_qoi(self, i):
    """
    Quantity of interest which integrates
    pressure over the boundary of the hole.
    """

    def steady_qoi():
        u, p = split(self.fields["up"])
        return p * ds(4)

    return steady_qoi
