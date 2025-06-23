from abc import ABC, abstractmethod

from firedrake import FunctionSpace, MeshHierarchy
from firedrake.petsc import PETSc

from goalie_adjoint import GoalOrientedMeshSeq, annotate_qoi


class ThetisMeshSeq(GoalOrientedMeshSeq):
    """
    A class to manage Thetis solver objects.
    """

    # thetis_model = None  # Default value for the Thetis model class
    # thetis_parameters = None  # Default value for Thetis parameters

    def __init__(
        self, time_partition, initial_meshes, thetis_model, thetis_parameters, **kwargs
    ):
        """
        Initialize the ThetisMeshSeq.

        :param thetis_model: An instance of BaseThetisModel or its derived class.
        :param thetis_parameters: Dictionary of Thetis parameters.
        """
        self.thetis_manager = {}
        self.thetis_model = thetis_model
        self.parameters = thetis_parameters

        # Initialize GoalOrientedMeshSeq
        super().__init__(time_partition, initial_meshes, **kwargs)

    def get_thetis_object(self, mesh):
        if mesh in self.thetis_manager.keys():
            # QC:
            # print("MESH IN MANAGER")
            return self.thetis_manager[mesh]

        else:
            # QC:
            # raise KeyError(f"Mesh {mesh} not found in the Thetis manager.")
            # print(f"Mesh {mesh} not found in the Thetis manager.")
            self.add_thetis_object(mesh)

            return self.thetis_manager[mesh]

    def add_thetis_object(self, mesh):
        # QC:
        # print(f' Add Thetis Object for: {mesh}')

        # get the goalie index for the current mesh:
        # QC:
        # print(f' start of add thetis object: meshes {self.meshes.index(mesh)}')
        mesh_index = self.meshes.index(mesh)

        # add the mesh to the dictionary
        # QC:
        # print(f"adding mesh to manager")

        # print(mesh, self.parameters)

        # create thetis model instance
        thetis_model_instance = self.thetis_model(mesh, self.parameters)

        # get TimePartition parameters for the current mesh
        thetis_model_instance.parameters["timestep"] = self.time_partition.timesteps[
            mesh_index
        ]
        # TODO: does this need to be synced up between Thetis and Goalie?
        # thetis_model_instance.parameters['t_export'] =\
        # self.time_partition.num_timesteps_per_export[mesh_index]
        # goalie t_end is exclusive
        thetis_model_instance.parameters["t_end"] = self.time_partition.subintervals[
            mesh_index
        ][1]  # - self.time_partition.timesteps[mesh_index]
        thetis_model_instance.parameters["t_start"] = self.time_partition.subintervals[
            mesh_index
        ][0]

        # add on 'spinup' if applicable - like a time bulk shift for forcing purposes
        if thetis_model_instance.parameters["t_spinup"] > 0.0:
            thetis_model_instance.parameters["t_start"] = (
                thetis_model_instance.parameters["t_start"]
                + thetis_model_instance.parameters["t_spinup"]
            )  # - self.params['timestep']
            thetis_model_instance.parameters["t_end"] = (
                thetis_model_instance.parameters["t_end"]
                + thetis_model_instance.parameters["t_spinup"]
            )  # - self.params['timestep']

        # QC:
        # print(thetis_model_instance.parameters)

        # initialize forcing function
        forcing = thetis_model_instance.set_update_forcings(
            thetis_model_instance.parameters["t_start"]
        )

        # QC:
        # print(f'forcing = {forcing(0)}')

        # initialize thetis solver2d object
        solv = thetis_model_instance.get_flowsolver2d(
            forcing(0.0), thetis_model_instance.initial_condition()
        )

        # get the QOI - take advantage of Thetis callbacks
        qoi = thetis_model_instance.calculate_qoi

        # need to match TimeStepper naming convention for adjoint dependencies
        solv.fields.solution_2d.rename("solution")

        # add to manager
        self.thetis_manager[mesh] = (solv, forcing, qoi)

        # QC:
        # print(f'added mesh to manager: full manager {self.thetis_manager}')

        # return self.thetis_manager[mesh]

    def get_thetis_solution(self, index):
        if self[index] in self.thetis_manager.keys():
            thetis_obj = self.get_thetis_object(self[index])[0]

            solution = thetis_obj.fields.solution_2d
            # QC:
            # print(f'\n\n thetis object exists, {index} mesh {self[index]}')
            return solution

        else:
            # _fs = self._initial_fs(index)
            thetis_model_instance = self.thetis_model(self[index], self.parameters)

            # QC:
            # print(f'\n\n first call of qoi, {index} mesh {self[index]}')

            return thetis_model_instance.initial_condition()

    def get_function_spaces(self, mesh):
        thetis_obj = self.get_thetis_object(mesh)[0]

        return {"solution": thetis_obj.fields.solution_2d.function_space()}

    def get_initial_condition(self):
        """
        Compute an initial condition based on the inflow velocity
        and zero free surface elevation.
        """

        thetis_obj = self.get_thetis_object(self[0])[0]

        return {"solution": thetis_obj.fields.solution_2d}

    def get_solver(self):
        def solver(index):
            # get associate thetis solver object
            thetis_obj = self.get_thetis_object(self[index])[0]
            update_forcings = self.get_thetis_object(self[index])[1]

            _sol, _sol_old = self.fields["solution"]  # current, last/ic

            thetis_obj.fields.solution_2d.assign(_sol_old)

            # Time integrate from t_start to t_end
            tp = self.time_partition
            t_start, t_end = tp.subintervals[index]
            dt = tp.timesteps[index]
            t = t_start
            thetis_obj.simulation_time = 0  # NEED THIS !!!!!

            qoi = self.get_qoi(index)

            # Communicate variational form to mesh_seq
            self.read_forms({"solution": thetis_obj.timestepper.F})

            thetis_timestepper = thetis_obj.create_iterator(
                update_forcings=update_forcings
            )

            while t < t_end - 1.0e-05:
                t_Thetis = next(thetis_timestepper)

                # QC:
                print(f"in solver: t {t} t_Thetis {t_Thetis}")

                if self.qoi_type == "time_integrated":
                    self.J += qoi(t)
                    # QC:
                    # print(f" in time_integrated qoi: {self.J}")

                # try assigning across before yield?
                _sol.assign(thetis_obj.fields.solution_2d)

                yield

                _sol_old.assign(_sol)
                t += dt

            # _sol.assign(thetis_obj.fields.solution_2d)

        return solver

    @annotate_qoi
    def get_qoi(self, i):
        """
        Idea is that the solver2d will have the solution
        as well as callbacks stored on it
        """
        thetis_flowsolver2d, _, calc_qoi_function = self.get_thetis_object(self[i])

        if self.qoi_type == "time_integrated":

            def time_integrated_qoi(t):
                return calc_qoi_function(thetis_flowsolver2d, self.qoi_type)

            return time_integrated_qoi
        else:
            # for both end_time and steady
            def qoi():
                return calc_qoi_function(thetis_flowsolver2d, self.qoi_type)

            return qoi

    @PETSc.Log.EventDecorator()
    def get_enriched_mesh_seq(self, enrichment_method="p", num_enrichments=1):
        """
        Override to pass additional arguments to constructor
        """
        if enrichment_method not in ("h", "p"):
            raise ValueError(f"Enrichment method '{enrichment_method}' not supported.")
        if num_enrichments <= 0:
            raise ValueError("A positive number of enrichments is required.")

        # Apply h-refinement
        if enrichment_method == "h":
            if any(mesh == self.meshes[0] for mesh in self.meshes[1:]):
                raise ValueError(
                    "h-enrichment is not supported for shallow-copied meshes."
                )
            meshes = [MeshHierarchy(mesh, num_enrichments)[-1] for mesh in self.meshes]
        else:
            meshes = self.meshes

        # Construct object to hold enriched spaces
        enriched_mesh_seq = type(self)(
            self.time_partition,
            meshes,
            thetis_model=self.thetis_model,
            thetis_parameters=self.parameters,
            get_function_spaces=self._get_function_spaces,
            get_initial_condition=self._get_initial_condition,
            get_solver=self._get_solver,
            get_qoi=self._get_qoi,
            qoi_type=self.qoi_type,
        )

        enriched_mesh_seq._update_function_spaces()

        # Apply p-refinement
        if enrichment_method == "p":
            for label, fs in enriched_mesh_seq.function_spaces.items():
                for n, _space in enumerate(fs):
                    element = _space.ufl_element()
                    element = element.reconstruct(
                        degree=element.degree() + num_enrichments
                    )
                    enriched_mesh_seq._fs[label][n] = FunctionSpace(
                        enriched_mesh_seq.meshes[n], element
                    )

        return enriched_mesh_seq


class BaseThetisModel(ABC):
    """
    Abstract Base Class for Thetis models. Defines the interface for Thetis models.
    """

    # Template for required parameters
    REQUIRED_PARAMETERS = set()

    def __init__(self, mesh, parameters):
        """
        Initialize the base Thetis model.

        :param mesh: Firedrake mesh object.
        :param parameters: Dictionary of simulation parameters.
        """
        self.mesh = mesh
        self.parameters = parameters.copy()

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate that all required parameters are present in the provided parameters.
        """
        missing_parameters = self.REQUIRED_PARAMETERS - self.parameters.keys()
        if missing_parameters:
            raise ValueError(f"Missing required parameters: {missing_parameters}")

    @abstractmethod
    def bathymetry(self):
        """
        Compute the bathymetry field for the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def initial_function_space():
        """
        Create the initial function space for the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def initial_condition(self):
        """
        Create the initial condition for the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def set_update_forcings(self, t_start):
        """
        Create a callback function to update time-dependent forcing fields.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_flowsolver2d(self, mesh, forcing, initial_condition, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the initial condition
        """

    @staticmethod
    @abstractmethod
    def calculate_qoi(flowsolver2d, qoi_type):
        """
        Create a callback function to update time-dependent forcing fields.
        Must be implemented by subclasses.
        """
        pass
