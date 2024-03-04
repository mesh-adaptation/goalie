"""
Driver functions for mesh-to-mesh data transfer.
"""
from .utility import assemble_mass_matrix, cofunction2function, function2cofunction
import firedrake
from firedrake.functionspaceimpl import WithGeometry
from firedrake.petsc import PETSc
from petsc4py import PETSc as petsc4py


__all__ = ["project"]


def project(source, target_space, **kwargs):
    r"""
    Overload :func:`firedrake.projection.project` to account for the case of two mixed
    function spaces defined on different meshes and for the adjoint projection operator
    when applied to :class:`firedrake.cofunction.Cofunction`\s.

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.

    :arg source: the function to be projected
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target_space: the function space which we seek to project into, or the function
        or cofunction to use as the target
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`,
        :class:`firedrake.function.Function`, or :class:`firedrake.cofunction.Cofunction`
    :returns: the projection
    :rtype: :class:`firedrake.function.Function`
    """
    if not isinstance(source, (firedrake.Function, firedrake.Cofunction)):
        raise NotImplementedError(
            "Can only currently project Functions and Cofunctions."
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
        return _project_adjoint(source, target, **kwargs)
    elif source.function_space() == target.function_space():
        return target.assign(source)
    else:
        return _project(source, target, **kwargs)


@PETSc.Log.EventDecorator()
def _project(source, target, **kwargs):
    """
    Apply mesh-to-mesh conservative projection.

    This function extends :func:`firedrake.projection.project`` to account for mixed
    spaces.

    :arg source: the Function to be projected
    :type source: :class:`firedrake.function.Function`
    :arg target: the Function which we seek to project onto
    :type target: :class:`firedrake.function.Function`.
    :returns: the target projection
    :rtype: :class:`firedrake.function.Function`

    Extra keyword arguments are passed to :func:`firedrake.projection.project``.
    """
    Vs = source.function_space()
    Vt = target.function_space()
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
    assert isinstance(target, firedrake.Function)
    if hasattr(Vt, "num_sub_spaces"):
        for s, t in zip(source.subfunctions, target.subfunctions):
            t.project(s, **kwargs)
    else:
        target.project(source, **kwargs)
    return target


@PETSc.Log.EventDecorator()
def _project_adjoint(target_b, source_b, **kwargs):
    """
    Apply an adjoint mesh-to-mesh conservative projection.

    The notation used here is in terms of the adjoint of standard projection.
    However, this function may also be interpreted as a projector in its own right,
    mapping ``target_b`` to ``source_b``.

    :arg target_b: seed cofunction from the target space of the forward projection
    :type target_b: :class:`firedrake.cofunction.Cofunction`
    :arg source_b: output cofunction from the source space of the forward projection
    :type source_b: :class:`firedrake.cofunction.Cofunction`.
    :returns: the adjoint projection
    :rtype: :class:`firedrake.cofunction.Cofunction`

    Extra keyword arguments are passed to :func:`firedrake.projection.project`.
    """
    from firedrake.supermeshing import assemble_mixed_mass_matrix

    # Map to Functions to apply the adjoint projection
    if not isinstance(target_b, firedrake.Function):
        target_b = cofunction2function(target_b)
    if not isinstance(source_b, firedrake.Function):
        source_b = cofunction2function(source_b)

    Vt = target_b.function_space()
    Vs = source_b.function_space()
    if hasattr(Vs, "num_sub_spaces"):
        if not hasattr(Vt, "num_sub_spaces"):
            raise ValueError(
                "Source space has multiple components but target space does not."
            )
        if Vs.num_sub_spaces() != Vt.num_sub_spaces():
            raise ValueError(
                "Inconsistent numbers of components in target and source spaces:"
                f" {Vs.num_sub_spaces()} vs. {Vt.num_sub_spaces()}."
            )
        target_b_split = target_b.subfunctions
        source_b_split = source_b.subfunctions
    elif hasattr(Vt, "num_sub_spaces"):
        raise ValueError(
            "Target space has multiple components but source space does not."
        )
    else:
        target_b_split = [target_b]
        source_b_split = [source_b]

    # Apply adjoint projection operator to each component
    if Vs == Vt:
        source_b.assign(target_b)
    else:
        for i, (t_b, s_b) in enumerate(zip(target_b_split, source_b_split)):
            ksp = petsc4py.KSP().create()
            ksp.setOperators(assemble_mass_matrix(t_b.function_space()))
            mixed_mass = assemble_mixed_mass_matrix(Vt[i], Vs[i])
            with t_b.dat.vec_ro as tb, s_b.dat.vec_wo as sb:
                residual = tb.copy()
                ksp.solveTranspose(tb, residual)
                mixed_mass.mult(residual, sb)  # NOTE: already transposed above

    # Map back to a Cofunction
    return function2cofunction(source_b)
