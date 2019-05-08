"""
Utilities for probability mass functions.
"""

from collections import Counter
from typing import Callable

import numpy as np

################
# Type Aliases #
################
PMF_FUNC = Callable[[np.ndarray], np.ndarray]


def compute_empirical_pmf(
        samples: np.ndarray
) -> PMF_FUNC:
    """
    Given a batch of samples of shape (num_samples, d), returns an empirical
    distribution: that is a function empirical_pmf(x: states) of signature:

        (np.ndarray ~ (num_states, d)) -> (np.ndarray ~ (num_states,))

    which maps states to their empirical probability.

    Args:
        samples (np.ndarray ~ (num_states, d)): A batch of samples on which
            to base the empirical probability.

    Returns:
        empirical_pmf function: A function mapping a batch of states to their
            empirical probabilities.
    """
    num_samples = samples.shape[0]
    counts = Counter(tuple(sample) for sample in samples)

    def empirical_pmf(x: np.ndarray) -> np.ndarray:
        count_vector = np.array([counts[tuple(state)] for state in x])
        empirical_probs = count_vector / num_samples
        return empirical_probs

    return empirical_pmf


def l2_distance_between_pmfs(
        all_states: np.ndarray,
        pmf_1: PMF_FUNC,
        pmf_2: PMF_FUNC
) -> float:
    """
    Computes the l2 distance between two probability mass functions.

    Args:
        all_states (np.ndarray ~ (num_states, d)): An array stacked state
            vectors.
        pmf_1 (function): A function which takes a set of vertically stacked
            state vectors and returns a stack of probabilities
        pmf_2 (function): A function which takes a set of vertically stacked
            state vectors and returns a stack of probabilities

    Returns:
        float: the l2-norm between the two probability mass functions.
    """
    pmf_1_array = pmf_1(all_states)
    pmf_2_array = pmf_2(all_states)
    return np.linalg.norm(pmf_1_array - pmf_2_array)