"""
Problem specification for a simple Burgers equation test case.

The test case is notable for Goalie because the prognostic equation is nonlinear.

Code here is based on that found at
    https://firedrakeproject.org/demos/burgers.py.html
"""

import ufl
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitSquareMesh

from goalie.field import Field

# Problem setup
n = 32
mesh = UnitSquareMesh(n, n, diagonal="left")
fields = [Field("uv_2d")]
end_time = 0.5
dt = 1 / n
dt_per_export = 2
steady = False
get_bcs = None


def get_function_spaces(mesh):
    r""":math:`\mathbb P2` space."""
    return {"uv_2d": VectorFunctionSpace(mesh, "CG", 2)}


def get_solver(self):
    """
    Burgers equation solved using a direct method and backward Euler timestepping.
    """

    def solver(i):
        t_start, t_end = self.time_partition.subintervals[i]
        dt = self.time_partition.timesteps[i]

        u, u_ = self.field_data["uv_2d"]

        # Setup variational problem
        dt = self.time_partition.timesteps[i]
        fs = self.function_spaces["uv_2d"][i]
        R = FunctionSpace(self[i], "R", 0)
        dtc = Function(R).assign(dt)
        nu = Function(R).assign(0.0001)
        v = TestFunction(fs)
        F = (
            ufl.inner((u - u_) / dtc, v) * ufl.dx
            + ufl.inner(ufl.dot(u, ufl.nabla_grad(u)), v) * ufl.dx
            + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )

        # Time integrate from t_start to t_end
        t = t_start
        qoi = self.get_qoi(i)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="uv_2d")
            if self.qoi_type == "time_integrated":
                self.J += qoi(t)
            yield

            u_.assign(u)
            t += dt

    return solver


def get_initial_condition(self):
    """
    Initial condition which is sinusoidal in the x-direction.
    """
    init_fs = self.function_spaces["uv_2d"][0]
    x, y = ufl.SpatialCoordinate(self.meshes[0])
    return {
        "uv_2d": Function(init_fs).interpolate(ufl.as_vector([ufl.sin(ufl.pi * x), 0]))
    }


def get_qoi(self, i):
    """
    Quantity of interest which computes the square :math:`L^2` norm over the right hand
    boundary.
    """
    R = FunctionSpace(self[i], "R", 0)
    dtc = Function(R).assign(self.time_partition.timesteps[i])

    def time_integrated_qoi(t):
        u = self.field_data["uv_2d"][0]
        return dtc * ufl.inner(u, u) * ufl.ds(2)

    def end_time_qoi():
        return time_integrated_qoi(end_time)

    if self.qoi_type == "end_time":
        dtc.assign(1.0)
        return end_time_qoi
    else:
        return time_integrated_qoi
