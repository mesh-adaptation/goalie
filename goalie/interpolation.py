"""
Driver functions for mesh-to-mesh data transfer.
"""

from .utility import assemble_mass_matrix, cofunction2function, function2cofunction
import firedrake
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py


__all__ = ["transfer", "interpolate", "project"]


@PETSc.Log.EventDecorator()
def transfer(source, target_space, transfer_method="project", **kwargs):
    r"""
    Overload functions :func:`firedrake.__future__.interpolate` and
    :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint interpolation
    operator when applied to :class:`firedrake.cofunction.Cofunction`\s.

    :arg source: the function to be transferred
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target_space: the function space which we seek to transfer onto, or the
        function or cofunction to use as the target
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`,
        :class:`firedrake.function.Function` or :class:`firedrake.cofunction.Cofunction`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project".
    :type transfer_method: str
    :returns: the transferred function
    :rtype: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
        :func:`firedrake.projection.project`.
    """
    if transfer_method not in ("interpolate", "project"):
        raise ValueError(
            f"Invalid transfer method: {transfer_method}."
            " Options are 'interpolate' and 'project'."
        )
    if not isinstance(source, (firedrake.Function, firedrake.Cofunction)):
        raise NotImplementedError(
            f"Can only currently {transfer_method} Functions and Cofunctions."
        )
    if isinstance(target_space, WithGeometry):
        target = firedrake.Function(target_space)
    elif isinstance(target_space, (firedrake.Cofunction, firedrake.Function)):
        target = target_space
    else:
        raise TypeError(
            "Second argument must be a FunctionSpace, Function, or Cofunction."
        )
    if isinstance(source, firedrake.Cofunction):
        return _transfer_adjoint(source, target, transfer_method, **kwargs)
    elif source.function_space() == target.function_space():
        return target.assign(source)
    else:
        return _transfer_forward(source, target, transfer_method, **kwargs)


@PETSc.Log.EventDecorator()
def interpolate(source, target_space, **kwargs):
    """
    A wrapper for :func:`transfer` with ``transfer_method="interpolate"``.
    """
    return transfer(source, target_space, transfer_method="interpolate", **kwargs)


@PETSc.Log.EventDecorator()
def project(source, target_space, **kwargs):
    """
    A wrapper for :func:`transfer` with ``transfer_method="interpolate"``.
    """
    return transfer(source, target_space, transfer_method="project", **kwargs)


@PETSc.Log.EventDecorator()
def _transfer_forward(source, target, transfer_method, **kwargs):
    """
    Apply mesh-to-mesh transfer operator to a Function.

    This function extends the functionality of :func:`firedrake.__future__.interpolate`
    and :func:`firedrake.projection.project` to account for mixed spaces.

    :arg source: the Function to be transferred
    :type source: :class:`firedrake.function.Function`
    :arg target: the Function which we seek to transfer onto
    :type target: :class:`firedrake.function.Function`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project".
    :type transfer_method: str
    :returns: the transferred Function
    :rtype: :class:`firedrake.function.Function`

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
        :func:`firedrake.projection.project`.
    """
    Vs = source.function_space()
    Vt = target.function_space()
    _validate_matching_spaces(Vs, Vt)
    assert isinstance(target, firedrake.Function)
    if hasattr(Vt, "num_sub_spaces"):
        for s, t in zip(source.subfunctions, target.subfunctions):
            if transfer_method == "interpolate":
                t.interpolate(s, **kwargs)
            elif transfer_method == "project":
                t.project(s, **kwargs)
            else:
                raise ValueError(
                    f"Invalid transfer method: {transfer_method}."
                    " Options are 'interpolate' and 'project'."
                )
    else:
        if transfer_method == "interpolate":
            target.interpolate(source, **kwargs)
        elif transfer_method == "project":
            target.project(source, **kwargs)
        else:
            raise ValueError(
                f"Invalid transfer method: {transfer_method}."
                " Options are 'interpolate' and 'project'."
            )
    return target


@PETSc.Log.EventDecorator()
def _transfer_adjoint(target_b, source_b, transfer_method, **kwargs):
    """
    Apply an adjoint mesh-to-mesh transfer operator to a Cofunction.

    :arg target_b: seed Cofunction from the target space of the forward projection
    :type target_b: :class:`firedrake.cofunction.Cofunction`
    :arg source_b: output Cofunction from the source space of the forward projection
    :type source_b: :class:`firedrake.cofunction.Cofunction`
    :kwarg transfer_method: the method to use for the transfer. Options are
        "interpolate" (default) and "project".
    :type transfer_method: str
    :returns: the transferred Cofunction
    :rtype: :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.__future__.interpolate` or
        :func:`firedrake.projection.project`.
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    # Map to Functions to apply the adjoint transfer
    if not isinstance(target_b, firedrake.Function):
        target_b = cofunction2function(target_b)
    if not isinstance(source_b, firedrake.Function):
        source_b = cofunction2function(source_b)

    Vt = target_b.function_space()
    Vs = source_b.function_space()
    if Vs == Vt:
        source_b.assign(target_b)
        return function2cofunction(source_b)

    _validate_matching_spaces(Vs, Vt)
    if hasattr(Vs, "num_sub_spaces"):
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint transfer operator to each component
    for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
        if transfer_method == "interpolate":
            s_b.interpolate(t_b, **kwargs)
        elif transfer_method == "project":
            ksp = petsc4py.KSP().create()
            ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
            mixed_mass = assemble_mixed_mass_matrix(Vt[i], Vs[i])
            with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
                residual = tb.copy()
                ksp.solveTranspose(tb, residual)
                mixed_mass.mult(residual, sb)  # NOTE: already transposed above
        else:
            raise ValueError(
                f"Invalid transfer method: {transfer_method}."
                " Options are 'interpolate' and 'project'."
            )

    # Map back to a Cofunction
    return function2cofunction(source_b)


def _validate_matching_spaces(Vs, Vt):
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                "Source space has multiple components but target space does not."
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                "Inconsistent numbers of components in source and target spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            "Target space has multiple components but source space does not."
        )
