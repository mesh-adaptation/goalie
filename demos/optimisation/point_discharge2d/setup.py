from firedrake import *
from goalie import *

__all__ = ["fields", "get_initial_condition", "get_solver", "get_qoi"]

fields = [
    Field("c", family="Lagrange", degree=1, unsteady=False),
    Field("u_x", family="Real", degree=0, unsteady=False, solved_for=False),
    Field("u_y", family="Real", degree=0, unsteady=False, solved_for=False),
    Field("D", family="Real", degree=0, unsteady=False, solved_for=False),
    Field("r", family="Real", degree=0, unsteady=False, solved_for=False),
    Field("yc", family="Real", degree=0, unsteady=False, solved_for=False),
]


def get_initial_condition(mesh_seq):
    P1 = mesh_seq.function_spaces["c"][0]
    R = mesh_seq.function_spaces["r"][0]
    return {
        "c": Function(P1),
        "u_x": Function(R).assign(1.0),
        "u_y": Function(R).assign(0.0),
        "D": Function(R).assign(0.1),
        "r": Function(R).assign(0.05211427),  # Calibrated radius parameter
        "yc": Function(R).assign(5.1),  # y-location parameter, to be optimised
    }


def point_source(mesh_seq, index, xloc, yloc):
    r = mesh_seq.field_functions["r"]
    x, y = SpatialCoordinate(mesh_seq[index])
    return 100.0 * exp(-((x - xloc) ** 2 + (y - yloc) ** 2) / r**2)


def get_solver(mesh_seq):
    def solver(index):
        function_space = mesh_seq.function_spaces["c"][index]

        # Define constants
        c = mesh_seq.field_functions["c"]
        u_x = mesh_seq.field_functions["u_x"]
        u_y = mesh_seq.field_functions["u_y"]
        u = as_vector([u_x, u_y])
        D = mesh_seq.field_functions["D"]

        # Define source
        x0, y0 = 2.0, 5.0
        xc, yc = 20.0, mesh_seq.field_functions["yc"]
        S = 0
        for xloc, yloc in [(x0, y0), (xc, yc)]:
            S += point_source(mesh_seq, index, xloc, yloc)

        # SUPG stabilisation parameter
        h = CellSize(mesh_seq[index])
        unorm = sqrt(dot(u, u))
        tau = 0.5 * h / unorm
        tau = min_value(tau, unorm * h / (6 * D))

        # Setup variational problem
        psi = TestFunction(function_space)
        psi = psi + tau * dot(u, grad(psi))
        F = (
            dot(u, grad(c)) * psi * dx
            + inner(D * grad(c), grad(psi)) * dx
            - S * psi * dx
        )
        bc = DirichletBC(function_space, 0, 1)

        solve(F == 0, c, bcs=bc, ad_block_tag="c")
        yield

    return solver


def get_qoi(mesh_seq, index):
    def qoi():
        c = mesh_seq.field_functions["c"]
        return -c * ds(2)

    return qoi
