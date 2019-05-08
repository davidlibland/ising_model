"""
A suite of non-deterministic pytest tests which ensure that the empirical
density generated from explict samples converges to the pmf.
"""

from src.polynomial_interactions_util import PMF
from src.tests.utils import (
    assert_convergence_of_sampling,
)


def test_convergence_of_exact_sampling():
    """
    Checks that the empirical distribution generated `PMF.sample_exactly`
    converges in the l2-norm to the analytical pmf.
    """
    assert_convergence_of_sampling(
        d=3,  # 3 Ising sites
        num_samples=10000,  # We take 1000 samples
        eps=1e-2,  # Convergence is to withing 1e-2
        sampler=PMF.sample_exactly
    )


def test_convergence_of_mcmc_sampling():
    """
    Checks that the empirical distribution generated `PMF.sample_exactly`
    converges in the l2-norm to the analytical pmf.
    """
    assert_convergence_of_sampling(
        d=3,  # 3 Ising sites
        num_samples=50000,  # We take 10000 samples
        eps=1e-1,  # Convergence is to withing 1e-1
        sampler=PMF.sample_Metropolis_Hastings
    )
