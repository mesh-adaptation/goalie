from thetis import *
from thetis.options import DiscreteTidalTurbineFarmOptions
from thetis.solver2d import FlowSolver2d

from goalie.field import Field

__all__ = ["fields", "get_initial_condition", "get_solver", "get_qoi"]


# Set up P1DGv-P1DG element
p1dg_element = FiniteElement(
    "Discontinuous Lagrange", triangle, 1, variant="equispaced"
)
p1dgv_element = VectorElement(p1dg_element, dim=2)
p1dgvp1dg_element = MixedElement([p1dgv_element, p1dg_element])
fields = [
    Field("solution_2d", finite_element=p1dgvp1dg_element, unsteady=False),
    Field("yc", family="Real", degree=0, unsteady=False, solved_for=False),
]


class TurbineSolver2d(FlowSolver2d):
    @unfrozen
    def __init__(self, mesh_seq, index, bathymetry, options=None):
        super().__init__(mesh_seq[index], bathymetry, options=options)
        self.mesh_seq = mesh_seq
        self.index = index

    def create_function_spaces(self):
        super().create_function_spaces()
        mesh_seq = self.mesh_seq
        self.function_spaces.V_2d = mesh_seq.function_spaces["solution_2d"][self.index]
        self.function_spaces.U_2d, self.function_spaces.H_2d = (
            self.function_spaces.V_2d.subspaces
        )

    def create_fields(self):
        super().create_fields()
        self.fields.solution_2d = self.mesh_seq.field_functions["solution_2d"]
        self.fields.uv_2d, self.fields.elev_2d = self.fields.solution_2d.subfunctions


def get_initial_condition(mesh_seq, init_control=260.0):
    solution_2d = Function(mesh_seq.function_spaces["solution_2d"][0])
    u, eta = solution_2d.subfunctions
    u.interpolate(as_vector((1e-03, 0.0)))
    eta.assign(0.0)
    yc = Function(mesh_seq.function_spaces["yc"][0]).assign(init_control)
    return {"solution_2d": solution_2d, "yc": yc}


def get_solver(mesh_seq):
    def solver(index):
        mesh = mesh_seq[index]
        u, eta = mesh_seq.field_functions["solution_2d"].subfunctions
        yc = mesh_seq.field_functions["yc"]

        # Specify bathymetry
        x, y = SpatialCoordinate(mesh)
        channel_depth = domain_constant(40.0, mesh)
        channel_width = domain_constant(500.0, mesh)
        bathymetry_scaling = domain_constant(2.0, mesh)
        P1_2d = get_functionspace(mesh, "CG", 1)
        y_prime = y - channel_width / 2
        bathymetry = Function(P1_2d)
        bathymetry.interpolate(
            channel_depth - (bathymetry_scaling * y_prime / channel_width) ** 2
        )

        # Setup solver
        solver_obj = TurbineSolver2d(mesh_seq, index, Constant(channel_depth))
        options = solver_obj.options
        options.element_family = "dg-dg"
        options.timestep = 1.0
        options.simulation_export_time = 1.0
        options.simulation_end_time = 0.5
        options.no_exports = True
        options.swe_timestepper_type = "SteadyState"
        options.swe_timestepper_options.solver_parameters = {
            "snes_rtol": 1.0e-12,
        }
        options.swe_timestepper_options.ad_block_tag = "solution_2d"
        # options.use_grad_div_viscosity_term = False
        options.horizontal_viscosity = Constant(0.5)
        options.quadratic_drag_coefficient = Constant(0.0025)
        # options.use_grad_depth_viscosity_term = False

        # Setup boundary conditions
        solver_obj.bnd_functions["shallow_water"] = {
            1: {"uv": Constant((3.0, 0.0))},
            2: {"elev": Constant(0.0)},
            3: {"un": Constant(0.0)},
            4: {"un": Constant(0.0)},
        }
        solver_obj.create_function_spaces()

        # Define the thrust curve of the turbine using a tabulated approach:
        # speeds_AR2000: speeds for corresponding thrust coefficients - thrusts_AR2000
        # thrusts_AR2000: list of idealised thrust coefficients of an AR2000 tidal
        # turbine using a curve fitting technique with:
        #   * cut-in speed = 1 m/s
        #   * rated speed = 3.05 m/s
        #   * cut-out speed = 5 m/s
        # (ramp up and down to cut-in and at cut-out speeds for model stability)
        # NOTE: Taken from Thetis:
        #    https://github.com/thetisproject/thetis/blob/master/examples/discrete_turbines/tidal_array.py
        speeds_AR2000 = [
            0.0,
            0.75,
            0.85,
            0.95,
            1.0,
            3.05,
            3.3,
            3.55,
            3.8,
            4.05,
            4.3,
            4.55,
            4.8,
            5.0,
            5.001,
            5.05,
            5.25,
            5.5,
            5.75,
            6.0,
            6.25,
            6.5,
            6.75,
            7.0,
        ]
        thrusts_AR2000 = [
            0.010531,
            0.032281,
            0.038951,
            0.119951,
            0.516484,
            0.516484,
            0.387856,
            0.302601,
            0.242037,
            0.197252,
            0.16319,
            0.136716,
            0.115775,
            0.102048,
            0.060513,
            0.005112,
            0.00151,
            0.00089,
            0.000653,
            0.000524,
            0.000442,
            0.000384,
            0.000341,
            0.000308,
        ]

        # Setup tidal farm
        farm_options = DiscreteTidalTurbineFarmOptions()
        turbine_density = Function(solver_obj.function_spaces.P1_2d).assign(1.0)
        farm_options.turbine_type = "table"
        farm_options.turbine_density = turbine_density
        farm_options.turbine_options.diameter = 18.0
        farm_options.turbine_options.thrust_speeds = speeds_AR2000
        farm_options.turbine_options.thrust_coefficients = thrusts_AR2000
        farm_options.upwind_correction = False
        farm_options.turbine_coordinates = [
            [domain_constant(456, mesh), domain_constant(250, mesh)],
            [domain_constant(456, mesh), domain_constant(310, mesh)],
            [domain_constant(456, mesh), domain_constant(190, mesh)],
            [domain_constant(744, mesh), yc],
        ]
        options.discrete_tidal_turbine_farms["everywhere"] = [farm_options]

        # Apply initial conditions and solve
        solver_obj.assign_initial_conditions(uv=u, elev=eta)
        solver_obj.iterate()

        # Stash info related to tidal farms
        mesh_seq.tidal_farm_options = farm_options
        assert len(solver_obj.tidal_farms) == 1
        mesh_seq.tidal_farm = solver_obj.tidal_farms[0]

        yield

    return solver


def get_qoi(mesh_seq, index):
    def steady_qoi():
        mesh = mesh_seq[index]
        u, eta = split(mesh_seq.field_functions["solution_2d"])
        yc = mesh_seq.field_functions["yc"]
        farm = mesh_seq.tidal_farm
        farm_options = mesh_seq.tidal_farm_options

        # Power output contribution
        J_power = farm.turbine.power(u, eta) * farm.turbine_density * ufl.dx

        # Add a regularisation term for constraining the control
        area = assemble(domain_constant(1.0, mesh) * ufl.dx)
        alpha = domain_constant(1.0 / area, mesh)
        y2 = farm_options.turbine_coordinates[1][1]
        y3 = farm_options.turbine_coordinates[2][1]
        J_reg = (
            alpha
            * ufl.conditional(
                yc < y3, (yc - y3) ** 2, ufl.conditional(yc > y2, (yc - y2) ** 2, 0)
            )
            * ufl.dx
        )

        # Sum the two components
        # NOTE: We rescale the functional such that the gradients are ~ order magnitude
        #       1
        # NOTE: We also multiply by -1 so that if we minimise the functional, we
        #       maximise power (maximize is also available from pyadjoint but currently
        #       broken)
        scaling = 10000
        return scaling * (-J_power + J_reg)

    return steady_qoi
