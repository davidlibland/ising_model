"""
Utility to do Metropolis Hastings sampling from a distribution.
"""

from typing import Iterator, Callable

import numpy as np


def get_batch_mh_sampler(
        initial_states: np.ndarray,
        energy_func: Callable[[np.ndarray], np.ndarray],
        transition_process: Callable[[np.ndarray], np.ndarray],
) -> Iterator[np.ndarray]:
    """
    Yields batches of mcmc samples using the metropolis hastings algorithm.
    States are assumed to be d-dimensional vectors, and the batch-size is
    determined by the shape of the initial_states vector: if initial_states
    is of shape (n, d), then the batch size is n.

    Notes:
        1. The n samples in a batch are computed by n parallel
    (and independent) mcmc chains, so all the samples in a given batch are
    mutually independent (but distinct batches are correlated).
        2. Samples are returned without any burn-in period. Be sure to discard
    a sufficient number of initial samples to approximate the true distribution.

    Args:
        initial_states (np.ndarray ~ (num_chains, d): A vector of initial
            states.
        energy_func: A function from states to energies, of signature
            (np.ndarray ~ (num_chains, d)) -> (np.ndarray ~ (num_chains,))
        transition_process: A random mixing process from states to states,
            of signature
            (np.ndarray ~ (num_chains, d)) -> (np.ndarray ~ (num_chains, d))

    Yields:
        (np.ndarray ~ (num_chains, d)), samples from the mcmc process.
    """
    states = initial_states
    num_chains = states.shape[0]
    energies = energy_func(states)
    while True:
        # Compute new candidate states:
        candidate_states = transition_process(states)
        new_energies = energy_func(candidate_states)
        energy_diffs = new_energies - energies
        # Compute the acceptance thresholds:
        acceptance_thresholds = np.exp(-energy_diffs)
        # We always accept candidate states with probability:
        # min(acceptance_thresholds, 1)
        uniform_samples = np.random.uniform(size=num_chains)
        acceptances = uniform_samples <= acceptance_thresholds
        # Update the state and energy vectors conditioned on acceptance:
        energies = np.where(acceptances, new_energies, energies)
        states = np.where(acceptances.reshape([-1, 1]), candidate_states, states)
        yield states
