class Solver:
    r"""
    Base class for solvers.
    """

    def get_function_spaces(self, mesh):
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_function_spaces'"
            " method."
        )

    def get_initial_condition(self, mesh_seq):
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_initial_condition'"
            " method."
        )

    def get_solver(self, mesh_seq, index):
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_solver' method."
        )

    def get_qoi(self, mesh_seq):
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_qoi' method."
        )
