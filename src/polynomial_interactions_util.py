"""
ML problem utilities
"""
import logging
from functools import lru_cache
from itertools import product, islice
from typing import Optional, Iterator

import numpy as np

from src.mcmc import get_batch_mh_sampler

"""
[Note you can pick your own representation for these antisymmetric forms]
"""

class PMF:
    """
    Implements a pmf over 3-dimensional Ising variables which is the
    canonical distribution for the energy function,
        H[x] = A(x,x) + B(x,x,x),
    in which A is an antisymmetric 2-form and B is an antisymmetric 3-form.

    Attributes:
        A (np.ndarray): an antisymmetric 2-form
        B (np.ndarray): an antisymmetric 3-form

    """
    def __init__(self, A: np.ndarray, B: np.ndarray):
        """
        Initializes the model parameters according to A, B.

        Args:
            A (np.ndarray): an antisymmetric 2-form
            B (np.ndarray): an antisymmetric 3-form

        """
        assert isinstance(A, np.ndarray), "A must be an array not %s" % type(A)
        assert isinstance(B, np.ndarray), "B must be an array not %s" % type(B)
        self._d = A.shape[0]
        assert tuple(A.shape) == (self._d, self._d), \
            "A must be a 2-form, got shape %s" % A.shape
        assert tuple(B.shape) == (self._d, self._d, self._d), \
            "B must be a 3-form, got shape %s" % B.shape
        self._A = A
        self._B = B
        self._mcmc_sampler = None

    def pmf(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the evaluation of the pmf on a batch of 3d Ising variables.

        Args:
            x (np.ndarray ~ (num_samples, 3)): batch of 3-d Ising variables

        Returns:
            pmf(x) (np.ndarray ~ (num_samples)): batch of probabilities

        """
        energies = self.energy(x)
        # Note: the partition function is cached, so it is only computed once:
        Z = self.partition_function()
        return np.exp(-energies)/Z

    def sample_exactly(self, shape=(1,)) -> np.ndarray:
        """
        Generates a tensor of random samples from the pmf of shape shape.  This
        function samples from the pmf exactly, i.e. without accept-reject
        criteria.

        Args:
            shape (tuple; optional): shape of sample array

        Returns:
            samples (np.ndarray ~ (*shape, 3)): tensor of 3d Ising variables
                drawn from the pmf.

        """
        all_states = self.all_states()
        all_probs = self.pmf(all_states)
        # ToDo: Optimize if necessary:
        # The first time `sample_exactly` is called, `self.pmf`
        # computes energy of each state twice: once to evaluate the partition
        # function, and a second time when computing the pmf for each state.
        # This is redundant. (Note: this isn't an issue for subsequent calls to
        # `sample_exactly`, which reuse the initial (cached) computation of the
        # partition function.
        sample_ixs = np.random.choice(len(all_states), size=shape, p=all_probs)
        return all_states[sample_ixs]

    def sample_Metropolis_Hastings(self, shape=(1,)) -> np.ndarray:
        """
        Generates a tensor of random samples from the pmf of shape shape.  This
        function samples from the pmf using the Metropolis-Hastings algorithm.

        Args:
            shape (tuple; optional): shape of sample array

        Returns:
            samples (np.ndarray ~ (*shape, 3)): tensor of 3d Ising variables
                drawn from the pmf.

        """
        # Compute the sample size:
        sample_size = int(np.prod(shape))
        # Take the corresponding number of samples:
        samples = list(islice(self.mcmc_sampler, sample_size))
        # Pack the samples into the desired shape:
        sample_array = np.vstack(samples)
        return sample_array.reshape((*shape, self._d))

    ###########################
    # Lower level API methods #
    ###########################

    def energy(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the energies of a batch of 3d Ising variables.

        Args:
            x (np.ndarray ~ (num_samples, 3)): batch of 3-d Ising variables

        Returns:
            energy(x) (np.ndarray ~ (num_samples)): batch of energies

        """
        pairwise_energy = np.einsum("jk,nj,nk->n", self._A, x, x)
        triplewise_energy = np.einsum("ijk,ni,nj,nk->n", self._B, x, x, x)
        return pairwise_energy+triplewise_energy

    def all_states(self) -> np.ndarray:
        """
        Returns all the states of the model, as an np.ndarray of shape
        (num_states, d), where d is the number of Ising sites, and
        num_states = 2^d is the number of states.

        Returns:
            (np.ndarray ~ (num_states, d)): all the states.

        """
        states = (-1, 1)
        all_states_iterator = product(*[states for _ in range(self._d)])
        all_states = np.array(list(all_states_iterator))
        return all_states

    @lru_cache()
    def partition_function(self, beta: Optional[float]=1) -> float:
        """
        Computes the partition function for the model.

        Notes: This computation is cached, so it is only computed once.

        Args:
            beta (Optional float): the inverse temperature, defaults to 1.

        Returns:
            partition(beta) (float): the value of the partition function at
            beta.

        """
        all_states = self.all_states()
        energies = beta*self.energy(all_states)
        return np.exp(-energies).sum()

    @property
    def mcmc_sampler(self) -> Iterator[np.ndarray]:
        """
        Yields individual samples of shape (d,) where d is the
        number of Ising sites.

        Yields:
            sample (np.ndarray ~ (d, )): tensor of d-dimensional Ising
                variable sampled using the Metropolis-Hastings algorithm.
        """
        if self._mcmc_sampler is None:
            self._mcmc_sampler = self.get_mcmc_sampler()
        return self._mcmc_sampler

    def get_mcmc_sampler(
            self,
            burn_in=1000,
            num_chains=10,
    ) -> Iterator[np.ndarray]:
        """
        Initializes and returns a new MCMC sampler.

        Args:
            burn_in: The number of burn-in samples to discard from each chain.
            num_chains: The number of parallel (independent) mcmc chains to
                cycle through when sampling. Samples within a window of
                size num_chains are mutually independent.

        Yields:
            sample (np.ndarray ~ (d, )): a d-dimensional Ising variable
                sampled using the Metropolis-Hastings algorithm. Here d is the
                number of Ising sites.

        """
        initial_states = np.random.choice([-1, 1], (num_chains, self._d))

        def mh_transition_process(x: np.ndarray) -> np.ndarray:
            """The transition process used in the metropolis hastings
            algorithm"""
            # Copy the initial state
            new_state = x.copy()
            for i in range(x.shape[0]):
                # For each state, flip the spin of a randomly selected site
                flipping_ix = np.random.choice(self._d)
                new_state[i, flipping_ix] *= -1
            return new_state

        batch_sampler = get_batch_mh_sampler(
            initial_states=initial_states,
            transition_process=mh_transition_process,
            energy_func=self.energy
        )
        # Be sure to burn the sampler in:
        for i in range(burn_in):
            next(batch_sampler)
            logging.info("Burn in: %d of %d" % (i + 1, burn_in))

        # yield individual samples:
        for batch in batch_sampler:
            for sample in batch:
                yield sample
