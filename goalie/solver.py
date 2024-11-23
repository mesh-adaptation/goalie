from abc import ABC, abstractmethod


class Solver(ABC):
    @abstractmethod
    def get_function_spaces(self, mesh):
        pass

    @abstractmethod
    def get_initial_condition(self, mesh_seq):
        pass

    @abstractmethod
    def get_solver(self, mesh_seq, index):
        pass

    def get_qoi(self, mesh_seq):
        return None
