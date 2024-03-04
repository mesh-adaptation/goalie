"""
Driver functions for metric-based mesh adaptation.
"""
from .log import debug
from animate.metric import RiemannianMetric
import firedrake
from firedrake.petsc import PETSc
import numpy as np
import ufl


__all__ = ["enforce_variable_constraints", "space_time_normalise", "ramp_complexity"]


@PETSc.Log.EventDecorator()
def enforce_variable_constraints(
    metrics,
    h_min=1.0e-30,
    h_max=1.0e30,
    a_max=1.0e5,
    boundary_tag=None,
):
    r"""
    Post-process a list of metrics to enforce minimum and maximum element sizes, as well
    as maximum anisotropy.

    :arg metrics: the metrics
    :type metrics: :class:`list` of :class:`~.RiemannianMetric`\s
    :kwarg h_min: minimum tolerated element size
    :type h_min: :class:`firedrake.function.Function`, :class:`float`, or :class:`int`
    :kwarg h_max: maximum tolerated element size
    :type h_max: :class:`firedrake.function.Function`, :class:`float`, or :class:`int`
    :kwarg a_max: maximum tolerated element anisotropy
    :type a_max: :class:`firedrake.function.Function`, :class:`float`, or :class:`int`
    :kwarg boundary_tag: optional tag to enforce sizes on.
    :type boundary_tag: :class:`str` or :class:`int`
    """
    from collections.abc import Iterable

    if isinstance(metrics, RiemannianMetric):
        metrics = [metrics]
    assert isinstance(metrics, Iterable)
    if not isinstance(h_min, Iterable):
        h_min = [h_min] * len(metrics)
    if not isinstance(h_max, Iterable):
        h_max = [h_max] * len(metrics)
    if not isinstance(a_max, Iterable):
        a_max = [a_max] * len(metrics)
    for metric, hmin, hmax, amax in zip(metrics, h_min, h_max, a_max):
        metric.enforce_variable_constraints(hmin, hmax, amax, boundary_tag=boundary_tag)
    return metrics


@PETSc.Log.EventDecorator()
def space_time_normalise(
    metrics,
    time_partition,
    metric_parameters,
    global_factor=None,
    boundary=False,
    restrict_sizes=True,
    restrict_anisotropy=True,
):
    r"""
    Apply :math:`L^p` normalisation in both space and time.

    Based on Equation (1) in :cite:`Barral:2016`.

    :arg metrics: the metrics associated with each subinterval
    :type metrics: :class:`list` of :class:`~.RiemannianMetric`\s
    :arg time_partition: temporal discretisation for the problem at hand
    :type time_partition: :class:`TimePartition`
    :arg metric_parameters: dictionary containing the target *space-time* metric
        complexity under `dm_plex_metric_target_complexity` and the normalisation order
        under `dm_plex_metric_p`, or a list thereof
    :type metric_parameters: :class:`list` of :class:`dict`\s or a single :class:`dict`
        to use for all subintervals
    :kwarg global_factor: pre-computed global normalisation factor
    :type global_factor: :class:`float`
    :kwarg boundary: if ``True``, the normalisation to be performed over the boundary
    :type boundary: :class:`bool`
    :kwarg restrict_sizes: if ``True``, minimum and maximum metric magnitudes are
        enforced
    :type restrict_sizes: :class:`bool`
    :kwarg restrict_anisotropy: if ``True``, maximum anisotropy is enforced
    :type restrict_anisotropy: :class:`bool`
    :returns: the space-time normalised metrics
    :rtype: :class:`list` of :class:`~.RiemannianMetric`\s
    """
    if isinstance(metric_parameters, dict):
        metric_parameters = [metric_parameters for _ in range(len(time_partition))]
    d = metrics[0].function_space().mesh().topological_dimension()
    if len(metrics) != len(time_partition):
        raise ValueError(
            "Number of metrics does not match number of subintervals:"
            f" {len(metrics)} vs. {len(time_partition)}."
        )
    if len(metrics) != len(metric_parameters):
        raise ValueError(
            "Number of metrics does not match number of sets of metric parameters:"
            f" {len(metrics)} vs. {len(metric_parameters)}."
        )

    # Preparation step
    metric_parameters = metric_parameters.copy()
    for metric, mp in zip(metrics, metric_parameters):
        if not isinstance(mp, dict):
            raise TypeError(
                "Expected metric_parameters to consist of dictionaries,"
                f" not objects of type '{type(mp)}'."
            )

        # Allow concise notation
        if "dm_plex_metric" in mp and isinstance(mp["dm_plex_metric"], dict):
            for key, value in mp["dm_plex_metric"].items():
                mp[f"dm_plex_metric_{key}"] = value
            mp.pop("dm_plex_metric")

        p = mp.get("dm_plex_metric_p")
        if p is None:
            raise ValueError("Normalisation order 'dm_plex_metric_p' must be set.")
        if not (np.isinf(p) or p >= 1.0):
            raise ValueError(
                f"Normalisation order '{p}' should be one or greater or np.inf."
            )
        target = mp.get("dm_plex_metric_target_complexity")
        if target is None:
            raise ValueError(
                "Target complexity 'dm_plex_metric_target_complexity' must be set."
            )
        if target <= 0.0:
            raise ValueError(f"Target complexity '{target}' is not positive.")
        metric.set_parameters(mp)
        metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

    # Compute global normalisation factor
    if global_factor is None:
        integral = 0
        p = mp["dm_plex_metric_p"]
        exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
        for metric, S in zip(metrics, time_partition):
            dX = (ufl.ds if boundary else ufl.dx)(metric.function_space().mesh())
            scaling = pow(S.num_timesteps, 2 * exponent)
            integral += scaling * firedrake.assemble(
                pow(ufl.det(metric), exponent) * dX
            )
        target = mp["dm_plex_metric_target_complexity"] * time_partition.num_timesteps
        debug(f"space_time_normalise: target space-time complexity={target:.4e}")
        global_factor = firedrake.Constant(pow(target / integral, 2 / d))
    debug(f"space_time_normalise: global scale factor={float(global_factor):.4e}")

    for metric, S in zip(metrics, time_partition):
        # Normalise according to the global normalisation factor
        metric.normalise(
            global_factor=global_factor,
            restrict_sizes=False,
            restrict_anisotropy=False,
        )

        # Apply the separate scale factors for each metric
        if not np.isinf(p):
            metric *= pow(S.num_timesteps, -2 / (2 * p + d))
        metric.enforce_spd(
            restrict_sizes=restrict_sizes,
            restrict_anisotropy=restrict_anisotropy,
        )

    return metrics


def ramp_complexity(base, target, iteration, num_iterations=3):
    """
    Ramp up the target complexity over the first few iterations.

    :arg base: the base complexity to start from
    :type base: :class:`float`
    :arg target: the desired complexity
    :type target: :class:`float`
    :arg iteration: the current iteration
    :type iteration: :class:`int`
    :kwarg num_iterations: how many iterations to ramp over?
    :type num_iterations: :class:`int`
    :returns: the ramped target complexity
    :rtype: :class:`float`
    """
    if base <= 0.0:
        raise ValueError(f"Base complexity must be positive, not {base}.")
    if target <= 0.0:
        raise ValueError(f"Target complexity must be positive, not {target}.")
    if iteration < 0:
        raise ValueError(f"Current iteration must be non-negative, not {iteration}.")
    if num_iterations < 0:
        raise ValueError(
            f"Number of iterations must be non-negative, not {num_iterations}."
        )
    alpha = 1 if num_iterations == 0 else min(iteration / num_iterations, 1)
    return alpha * target + (1 - alpha) * base
