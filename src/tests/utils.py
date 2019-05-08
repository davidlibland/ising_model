"""
Utilities for the tests.
"""
from typing import Optional, Callable, Tuple

import numpy as np

from src.pmf_utils import compute_empirical_pmf, \
    l2_distance_between_pmfs
from src.polynomial_interactions_util import PMF


def get_random_polynomial_interaction_class(d: Optional[int]=3) -> PMF:
    """Generate a random instance of :class:`PMF`: The interaction strengths
    are randomly sampled from the standard normal distribution."""
    A = np.random.randn(d, d)
    B = np.random.randn(d, d, d)
    return PMF(A, B)


def assert_convergence_of_sampling(
        num_samples: Optional[int]=100,
        d: Optional[int]=3,
        eps: Optional[float]=1e-2,
        sampler: Optional=None
):
    """
    Checks that after num_samples samples, the l2 distance between the empirical
    mass function and the analytic mass function is less than eps.

    Args:
        num_samples (int): the number of samples to base the estimate on.
        d (int; optional): the dimension of binary vector (defaults to 3).
        eps (float; optional): the threshold (defaults to 1e-2).
        sampler (function): the sampling method (defaults to `sample_exactly`)
    """
    distance = estimate_empirical_distance(num_samples, d, sampler)
    assert distance < eps, \
        "Expected the l2-distance, %s, between the empirical and analytic " \
        "mass functions be less than %s." % (distance, eps)


def estimate_empirical_distance(num_samples=10, d=3, sampler: Optional=None):
    """
    Computes the l2 distance between the empirical density and the analytic
    mass functions after num_samples samples.

    Args:
        num_samples (int): the number of samples to base the estimate on.
        d (int; optional): the dimension of binary vector (defaults to 3).
        sampler (function): the sampling method (defaults to `sample_exactly`)

    Returns:
        float: the l2-norm between the emprical and the analytic mass functions.
    """
    if not sampler:
        sampler = PMF.sample_exactly
    density_object = get_random_polynomial_interaction_class(d)
    return estimate_empirical_distance_for_instance(
        density_object=density_object,
        num_samples=num_samples,
        sampler=sampler
    )


def estimate_empirical_distance_for_instance(
        density_object: PMF,
        num_samples: int,
        sampler: Callable[[PMF, Tuple], np.ndarray]
):
    """
    Computes the l2 distance between the empirical density and the analytic
    mass functions after num_samples samples.

    Args:
        density_object (PMF): the density object (of class:`PMF`) to analyze.
        num_samples (int): the number of samples to base the estimate on.
        sampler (function): the sampling method (defaults to `sample_exactly`)

    Returns:
        float: the l2-norm between the emprical and the analytic mass functions.
    """
    samples = sampler(density_object, (num_samples,))
    empirical_pmf = compute_empirical_pmf(samples)
    all_states = density_object.all_states()
    return l2_distance_between_pmfs(
        all_states=all_states,
        pmf_1=empirical_pmf,
        pmf_2=density_object.pmf
    )
