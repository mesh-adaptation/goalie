"""
Test adjoint drivers.
"""

import importlib
import os
import sys
import unittest

import pyadjoint
import pytest
from animate.utility import errornorm, norm
from firedrake import Cofunction, UnitTriangleMesh

from goalie_adjoint import *

sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))

# ---------------------------
# unit tests
# ---------------------------


class TestAdjointMeshSeqGeneric(unittest.TestCase):
    """
    Generic unit tests for :class:`AdjointMeshSeq`.
    """

    def setUp(self):
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])
        self.meshes = [UnitTriangleMesh()]

    def test_qoi_type_error(self):
        with self.assertRaises(ValueError) as cm:
            AdjointMeshSeq(self.time_interval, self.meshes, qoi_type="blah")
        msg = (
            "QoI type 'blah' not recognised. "
            "Choose from 'end_time', 'time_integrated', or 'steady'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_get_qoi_notimplemented_error(self):
        mesh_seq = AdjointMeshSeq(self.time_interval, self.meshes, qoi_type="end_time")
        with self.assertRaises(NotImplementedError) as cm:
            mesh_seq.get_qoi({}, 0)
        msg = "'get_qoi' is not implemented."
        self.assertEqual(str(cm.exception), msg)


# ---------------------------
# standard tests for pytest
# ---------------------------

all_problems = [
    "point_discharge2d",
    "point_discharge3d",
    "steady_flow_past_cyl",
    "burgers",
]


@pytest.fixture(params=all_problems)
def problem(request):
    return request.param


@pytest.fixture(
    params=[
        "end_time",
        "time_integrated",
    ]
)
def qoi_type(request):
    return request.param


@pytest.mark.slow
def test_adjoint_same_mesh(problem, qoi_type, debug=False):
    """
    Check that `solve_adjoint` gives the same
    result when applied on one or two subintervals.

    :arg problem: string denoting the test case of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    :kwarg debug: toggle debugging mode
    """
    if debug:
        set_log_level(DEBUG)

    # Imports
    pyrint(f"\n--- Setting up {problem} test case with {qoi_type} QoI\n")
    test_case = importlib.import_module(problem)
    end_time = test_case.end_time
    steady = test_case.steady
    if "solid_body_rotation" in problem:
        end_time /= 4  # Reduce testing time
    elif steady and qoi_type == "time_integrated":
        pytest.skip("n/a for steady case")
    if steady:
        assert test_case.dt_per_export == 1
        assert np.isclose(end_time, test_case.dt)
        qoi_type = "steady"

    # Partition time interval and create MeshSeq
    time_partition = TimeInterval(
        end_time,
        test_case.dt,
        test_case.fields,
        num_timesteps_per_export=test_case.dt_per_export,
    )
    mesh_seq = AdjointMeshSeq(
        time_partition,
        test_case.mesh,
        get_function_spaces=test_case.get_function_spaces,
        get_initial_condition=test_case.get_initial_condition,
        get_form=test_case.get_form,
        get_solver=test_case.get_solver,
        get_qoi=test_case.get_qoi,
        qoi_type=qoi_type,
    )

    # Solve forward and adjoint without solve_adjoint
    pyrint("\n--- Adjoint solve on 1 subinterval using pyadjoint\n")
    if not pyadjoint.annotate_tape():
        pyadjoint.continue_annotation()
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    ic = mesh_seq.initial_condition
    controls = [pyadjoint.Control(value) for key, value in ic.items()]
    sols = mesh_seq.solver(0, ic)
    qoi = mesh_seq.get_qoi(sols, 0)
    J = mesh_seq.J if qoi_type == "time_integrated" else qoi()
    m = pyadjoint.enlisting.Enlist(controls)
    assert pyadjoint.annotate_tape()
    pyadjoint.pause_annotation()
    with tape.marked_nodes(m):
        tape.evaluate_adj(markings=True)
    # FIXME: Using mixed Functions as Controls not correct
    J_expected = float(J)

    # Get expected adjoint solutions and values at the timestep corresponding
    # to the first exported solution in the MeshSeq solution data
    first_export_idx = test_case.dt_per_export - 1
    adj_sols_expected = {}
    adj_values_expected = {}
    for field, fs in mesh_seq._fs.items():
        solve_blocks = mesh_seq.get_solve_blocks(field, 0)
        adj_sols_expected[field] = solve_blocks[first_export_idx].adj_sol.copy(
            deepcopy=True
        )
        if not steady:
            dep = mesh_seq._dependency(field, 0, solve_blocks[first_export_idx])
            adj_values_expected[field] = Cofunction(fs[0].dual())
            adj_values_expected[field].assign(dep.adj_value)

    # Loop over having one or two subintervals
    for N in range(1, 2 if steady else 3):
        pl = "" if N == 1 else "s"
        pyrint(f"\n--- Adjoint solve on {N} subinterval{pl} using goalie\n")

        # Solve forward and adjoint on each subinterval
        time_partition = TimePartition(
            end_time,
            N,
            test_case.dt,
            test_case.fields,
            num_timesteps_per_export=test_case.dt_per_export,
        )
        mesh_seq = AdjointMeshSeq(
            time_partition,
            test_case.mesh,
            get_function_spaces=test_case.get_function_spaces,
            get_initial_condition=test_case.get_initial_condition,
            get_form=test_case.get_form,
            get_solver=test_case.get_solver,
            get_qoi=test_case.get_qoi,
            qoi_type=qoi_type,
        )
        solutions = mesh_seq.solve_adjoint(
            get_adj_values=not steady, test_checkpoint_qoi=True
        )

        # Check quantities of interest match
        if not np.isclose(J_expected, mesh_seq.J):
            raise ValueError(f"QoIs do not match ({J_expected} vs. {mesh_seq.J})")

        # Check adjoint solutions at first export time match
        first_export_time = test_case.dt * test_case.dt_per_export
        for field in time_partition.fields:
            adj_sol_expected = adj_sols_expected[field]
            expected_norm = norm(adj_sol_expected)
            if np.isclose(expected_norm, 0.0):
                raise ValueError(
                    f"'Expected' norm at t={first_export_time} is unexpectedly zero."
                )
            adj_sol_computed = solutions[field].adjoint[0][0]
            err = errornorm(adj_sol_expected, adj_sol_computed) / expected_norm
            if not np.isclose(err, 0.0):
                raise ValueError(
                    f"Adjoint solutions do not match at t={first_export_time}"
                    f" (error {err:.4e}.)"
                )

        # Check adjoint actions at first export time match
        if not steady:
            for field in time_partition.fields:
                adj_value_expected = adj_values_expected[field]
                adj_value_computed = solutions[field].adj_value[0][0]
                err = errornorm(adj_value_expected, adj_value_computed) / norm(
                    adj_value_expected
                )
                if not np.isclose(err, 0.0):
                    raise ValueError(
                        f"Adjoint values do not match at t={first_export_time}"
                        f" (error {err:.4e}.)"
                    )

    tape = pyadjoint.get_working_tape()
    tape.clear_tape()


def plot_solutions(problem, qoi_type, debug=True):
    """
    Plot the forward and adjoint solutions, their lagged
    counterparts and the adjoint values corresponding to
    each field and exported timestep.

    :arg problem: string denoting the test case of choice
    :arg qoi_type: is the QoI evaluated at the end time
        or as a time integral?
    :kwarg debug: toggle debugging mode
    """
    import firedrake_adjoint  # noqa

    if debug:
        set_log_level(DEBUG)

    test_case = importlib.import_module(problem)
    end_time = test_case.end_time
    steady = test_case.steady
    time_partition = TimeInterval(
        end_time,
        test_case.dt,
        test_case.fields,
        num_timesteps_per_export=test_case.dt_per_export,
    )
    solutions = AdjointMeshSeq(
        time_partition,
        test_case.mesh,
        get_function_spaces=test_case.get_function_spaces,
        get_initial_condition=test_case.get_initial_condition,
        get_form=test_case.get_form,
        get_solver=test_case.get_solver,
        get_qoi=test_case.get_qoi,
        qoi_type=qoi_type,
    ).solve_adjoint(get_adj_values=not steady, test_checkpoint_qoi=True)
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", problem)
    outfiles = AttrDict(
        {
            "forward": VTKFile(os.path.join(output_dir, "forward.pvd")),
            "forward_old": VTKFile(os.path.join(output_dir, "forward_old.pvd")),
            "adjoint": VTKFile(os.path.join(output_dir, "adjoint.pvd")),
        }
    )
    if not steady:
        outfiles.adjoint_next = VTKFile(os.path.join(output_dir, "adjoint_next.pvd"))
        outfiles.adj_value = VTKFile(os.path.join(output_dir, "adj_value.pvd"))
    for label in outfiles:
        for k in range(time_partition.num_exports_per_subinterval[0] - 1):
            to_plot = []
            for field in time_partition.fields:
                sol = solutions[field][label][0][k]
                to_plot += (
                    [sol]
                    if not hasattr(sol, "subfunctions")
                    else list(sol.subfunctions)
                )
            outfiles[label].write(*to_plot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="test/test_adjoint.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("problem", type=str)
    parser.add_argument("qoi_type", type=str)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    assert args.qoi_type in ("end_time", "time_integrated", "steady")
    assert args.problem in all_problems
    if args.plot:
        plot_solutions(args.problem, args.qoi_type, debug=True)
    else:
        test_adjoint_same_mesh(args.problem, args.qoi_type, debug=True)
