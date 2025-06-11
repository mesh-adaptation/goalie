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
    def __init__(self, mesh_seq, bathymetry, options=None):
        super().__init__(mesh_seq[0], bathymetry, options=options)
        self.mesh_seq = mesh_seq

    def create_function_spaces(self):
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, "DG", 0)
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, "CG", 1)
        self.function_spaces.P1v_2d = get_functionspace(
            self.mesh2d, "CG", 1, vector=True
        )
        self.function_spaces.V_2d = self.mesh_seq.function_spaces["solution_2d"][0]
        self.function_spaces.U_2d, self.function_spaces.H_2d = (
            self.function_spaces.V_2d.subspaces
        )
        self.function_spaces.P1DGv_2d = self.function_spaces.U_2d
        self.function_spaces.P1DG_2d = self.function_spaces.H_2d
        self.function_spaces.Q_2d = self.function_spaces.P1DG_2d

    def create_fields(self):
        if not hasattr(self.function_spaces, "U_2d"):
            self.create_function_spaces()

        if self.options.log_output and not self.options.no_exports:
            mode = "a" if self.keep_log else "w"
            set_log_directory(self.options.output_directory, mode=mode)

        # Add general fields
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)
        self.set_wetting_and_drying_alpha()
        self.depth = DepthExpression(
            self.fields.bathymetry_2d,
            use_nonlinear_equations=self.options.use_nonlinear_equations,
            use_wetting_and_drying=self.options.use_wetting_and_drying,
            wetting_and_drying_alpha=self.options.wetting_and_drying_alpha,
        )

        # Add fields for shallow water modelling
        self.fields.solution_2d = self.mesh_seq.field_functions["solution_2d"]
        uv_2d, elev_2d = (
            self.fields.solution_2d.subfunctions
        )  # correct treatment of the split 2d functions
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d

        # Add tracer fields
        self.solve_tracer = len(self.options.tracer_fields.keys()) > 0
        if self.solve_tracer:
            raise NotImplementedError

        # Add suspended sediment field
        if self.options.sediment_model_options.solve_suspended_sediment:
            raise NotImplementedError

        # Add fields for non-hydrostatic mode
        if self.options.nh_model_options.solve_nonhydrostatic_pressure:
            raise NotImplementedError


def get_initial_condition(mesh_seq, init_control=260.0):
    x, y = SpatialCoordinate(mesh_seq[0])
    solution_2d = Function(mesh_seq.function_spaces["solution_2d"][0])
    u, eta = solution_2d.subfunctions
    u.interpolate(as_vector((1e-03, 0.0)))
    eta.assign(0.0)
    yc = Function(mesh_seq.function_spaces["yc"][0]).assign(init_control)
    return {"solution_2d": solution_2d, "yc": yc}


def get_solver(mesh_seq):
    def solver(index):
        mesh = mesh_seq[index]
        solution_2d = mesh_seq.field_functions["solution_2d"]
        u, eta = solution_2d.subfunctions
        yc = mesh_seq.field_functions["yc"]

        # Specify bathymetry
        x, y = SpatialCoordinate(mesh_seq[0])
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
        solver_obj = TurbineSolver2d(mesh_seq, Constant(channel_depth))
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
            [456, 250],
            [456, 310],
            [456, 190],
            [744, yc],
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
        mesh = mesh_seq[0]
        u, eta = mesh_seq.field_functions["solution_2d"].subfunctions
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
