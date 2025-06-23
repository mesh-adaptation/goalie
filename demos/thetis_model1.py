# from thetis > demos > 01-2d-channel.py

import firedrake as fd
import numpy as np
import thetis as thetis

# to get delta_x only
from goalie_adjoint import *


class ThetisModel1(BaseThetisModel):
    """
    Abstract Base Class for Thetis models. Defines the interface for Thetis models.
    """

    REQUIRED_PARAMETERS = {
        "depth",
        "t_export",
        "t_end",
        "t_start",
        "timestep",
        "tide_amplitude",
        "tide_period",
        "output_directory",
    }

    def __init__(self, mesh, parameters):
        super().__init__(mesh, parameters)

    # @staticmethod
    def initial_function_space(self):
        """
        Create the initial function space for the model.
        Static function
        """
        P1v_2d = thetis.get_functionspace(self.mesh, "DG", 1, vector=True)
        P1_2d = thetis.get_functionspace(self.mesh, "CG", 1)
        _ifs = P1v_2d * P1_2d

        # QC:
        # print(f'\n in intial fs, mesh {self.mesh}')

        return _ifs

    # @staticmethod
    def initial_condition(self):
        """
        Create the initial condition for the model.
        Must be implemented by subclasses.
        """
        # QC:
        # print(f"\n\n initialize from initial conditions  mesh {self.mesh}")

        # define initial condition function from initial function space
        _ic = fd.Function(self.initial_function_space())
        vel_init, elev_init = _ic.subfunctions  # fd.split(q) # avoid subfunctions?

        # QC:
        # print(f"get initial conditions {field}")

        return _ic

    def bathymetry(self):
        """
        Compute the bathymetry field on the current `mesh`.
        """
        # NOTE: We assume a constant bathymetry field
        P1_2d = thetis.get_functionspace(self.mesh, "CG", 1)
        return fd.Function(P1_2d, name="Bathymetry").assign(self.parameters["depth"])

    def tidal_elevation(self, simulation_time):
        """Time-dependent tidal elevation"""

        tide_amp = self.parameters["tide_amplitude"]  # amplitude of the tide
        tide_t = self.parameters["tide_period"]  # period of the tide

        elev = tide_amp * fd.sin(2 * fd.pi * simulation_time / tide_t)

        # QC:
        # print(f'\n ELEV: {elev} sim time: {simulation_time}')

        return elev

    def set_update_forcings(self, t_start):
        """Callback function that updates all time dependent forcing fields"""

        # index_ = self.meshes.index(self.mesh)
        R = fd.FunctionSpace(self.mesh, "R", 0)

        # QC:
        # print(f'\n\n in set update forcings, t_start {t_start}')

        tide_elev_const = fd.Function(R).assign(self.tidal_elevation(t_start))

        # qoi = self.get_qoi(index_)

        def forcings(t_new):
            # QC:
            # logging.info(f"time: {t_new+t_start}")

            tide_elev_const.assign(self.tidal_elevation(t_new + t_start))

            # QC:
            # print(f"UPDATE FORCING {t_new+t_start}: {tide_elev_const.dat.data[:]}")

            return tide_elev_const

        return forcings

    def get_flowsolver2d(self, forcing, initial_condition, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the initial condition
        """
        print(f"flow solver get: {self.parameters}")

        # define the bathymetry
        bathymetry = self.bathymetry()

        # Create solver object
        thetis_solver = thetis.solver2d.FlowSolver2d(self.mesh, bathymetry)
        options = thetis_solver.options

        # QC:
        # print(f"\n\n in get flowsolver index {self.mesh}\
        #        - {self.parameters['t_start']}, {self.parameters['t_end']}")

        options.output_directory = self.parameters["output_directory"]
        options.fields_to_export = ["elev_2d", "uv_2d", "Bathymetry"]
        # options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d','Bathymetry']

        # time stepper options  ej321 - add for understanding
        options.timestep = self.parameters["timestep"]
        print(self.parameters["t_export"])
        options.simulation_export_time = self.parameters["t_export"]
        # try to pass with dt buffer for end time StopIteration
        options.simulation_end_time = (
            self.parameters["t_end"] - self.parameters["t_start"]
        )  # -self.parameters['timestep']

        # QC:
        # print(self.parameters['t_start'], self.parameters['t_end'])

        options.swe_timestepper_type = "CrankNicolson"

        options.swe_timestepper_options.ad_block_tag = "solution"  # for adjoint

        R = fd.FunctionSpace(self.mesh, "R", 0)

        thetis_solver.bnd_functions["shallow_water"] = {
            1: {"elev": fd.Function(R).assign(0.0)},
            2: {"elev": forcing},
        }

        # Apply initial guess
        # u_init, eta_init = ic.subfunctions
        vel_init, elev_init = initial_condition.subfunctions
        thetis_solver.assign_initial_conditions(uv=vel_init, elev=elev_init)

        # parameter options ej321 - add for understanding
        options.no_exports = False
        options.update(kwargs)
        thetis_solver.create_equations()

        # return solver objected
        return thetis_solver

    @staticmethod
    def calculate_qoi(flowsolver2d, qoi_type):
        solution = flowsolver2d.fields.solution_2d
        # QC:
        # print(f'in calc_qoi: type = {qoi_type}')
        _velocity, _eta = fd.split(solution)

        if qoi_type == "steady":
            # QC:
            # print(f"\n in steady qoi")
            return fd.inner(_eta, _eta) * fd.dx
        if qoi_type == "end_time":
            # QC:
            # print(f"\n in end time qoi")
            return fd.inner(_eta, _eta) * fd.ds(2)
        if qoi_type == "time_integrated":
            # QC:
            # print(f"\n in time integrated qoi")
            dt = flowsolver2d.dt
            return dt * fd.inner(_eta, _eta) * fd.ds(2)


if __name__ == "__main__":
    # define mesh
    nx, ny = 25, 2
    L, W = 40e3, 2e3
    num_meshes = 2
    meshes = []
    # this works to generate independent meshes per partition
    if num_meshes > 1:
        for m in range(num_meshes):
            meshes.append(fd.RectangleMesh(nx, ny, L, W, name=f"mesh_0_{m}"))
    else:
        meshes.append(fd.RectangleMesh(nx, ny, L, W, name="mesh_0_0"))

    print(f"\n\n INITAL meshes: {meshes}")

    # define time segmentation
    thetis_parameters = {
        "depth": 20.0,
        "tide_amplitude": 0.01,
        "tide_period": 12 * 3600,
        "t_end": 12 * 3600,
        "t_start": 0.0,
        "t_export": 1200.0,
        "timestep": 200.0,
        "t_spinup": 0.0,
        "output_directory": "outputs_2d_channel",
        "qoi_type": "end_time",
    }

    n_timesteps_per = int(
        np.floor(thetis_parameters["t_export"] / thetis_parameters["timestep"]) or 1
    )

    time_partition = TimePartition(
        end_time=thetis_parameters["t_end"],
        num_subintervals=num_meshes,
        timesteps=thetis_parameters["timestep"],
        field_names=["solution"],  # this needs to be called 'solution'
        num_timesteps_per_export=n_timesteps_per,
    )

    print(f"TIME PARTITION: {time_partition}")
    # define mesh sequence instance
    mesh_seq = ThetisMeshSeq(
        time_partition,
        meshes,
        ThetisModel1,
        thetis_parameters,
        qoi_type="time_integrated",
    )
    print(mesh_seq.meshes)

    # run forward
    solutions = mesh_seq.solve_forward()

    print(solutions)

    # solutions, indicators = mesh_seq.indicate_errors(
    #     enrichment_kwargs={"enrichment_method": "h"}
    # )

    # solutions = mesh_seq.solve_adjoint()

    # Plotting this, we find that the results are consistent with those generated
    # previously. ::

    # fig, axes, tcs = plot_snapshots(
    #     solutions, time_partition, "solution", "forward", levels=np.linspace(0, 1, 9)
    # )

    # .. figure:: burgers-oo_ee.jpg
    #    :figwidth: 90%
    #    :align: center

    # fig, axes, tcs = plot_snapshots(solutions, time_partition, "solution", "adjoint")
    # fig.savefig("thetis_adjoint.jpg")

    # .. figure:: burgers-oo-time_integrated.jpg
    #    :figwidth: 90%
    #    :align: center

    # In the `next demo <./solid_body_rotation.py.html>`__, we move on from
    # to consider a linear advection example with a rotational velocity field.
    #
    # This demo can also be accessed as a `Python script <burgers_oo.py>`__.
