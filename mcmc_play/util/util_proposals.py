import numpy as np
from numpy.linalg import inv
import random


# Make a proposal distribution
def proposal_distr(vec):
    std_proposal = 0.05
    return vec + std_proposal * np.random.randn(len(vec))


def gibbs_proposal(vec, mean, cov):
    assert len(vec) == len(mean) == cov.shape[0]
    num_dim = len(vec)

    # Sample a random dimension from the total number of dimensions
    dim = np.random.randint(0, num_dim)

    # Sample a new value for that dim, conditioned on all other variables
    # Condition the gaussian for dim on the observed values for all other dims
    # Using equation 353 from chapter 8 of the Matrix Cook Book
    mu_a = np.delete(mean, dim)
    mu_b = mean[dim]
    vec_a = np.delete(vec, dim)

    cov_a = np.delete(np.delete(cov, dim, axis=0), dim, axis=1)
    cov_c = np.delete(cov[:, dim], dim)
    cov_b = cov[dim, dim]

    gamma = cov_c.T.dot(inv(cov_a))
    cond_mean = mu_b + np.inner(gamma, vec_a - mu_a)
    cond_cov = cov_b - np.inner(gamma, cov_c)

    vec[dim] = cond_mean + np.sqrt(cond_cov) * np.random.randn()
    return vec


# An iterator for the MCMC chain
def random_walk_chain(num_iters, dim, distro):
    # Start with some initial vector
    current_sample = np.zeros((dim,))

    # Also track the acceptance rate
    total_accepts = 0

    for num_iter in range(num_iters):
        # For every step, we make a new proposal
        proposed_sample = proposal_distr(current_sample)

        # And then accept it according to the Metropolis Hastings rule
        alpha = distro.pdf(proposed_sample) / distro.pdf(current_sample)

        if alpha >= 1.0:
            accept = True
            current_sample = np.copy(proposed_sample)
        else:
            accept = random.random() < alpha
            if accept:
                current_sample = np.copy(proposed_sample)
        total_accepts += accept
        yield current_sample, total_accepts / (num_iter + 1)


def gibbs_chain(num_iters, dim, distro):
    # Start with some initial vector
    current_sample = np.zeros((dim,))

    # Also track the acceptance rate
    total_accepts = 0

    for num_iter in range(num_iters):
        # For every step, we make a new proposal
        proposed_sample = gibbs_proposal(current_sample, distro.mean, distro.cov)

        # For Gibbs sampling, the Metropolis Hastings acceptance step is always one :)
        current_sample = np.copy(proposed_sample)

        total_accepts += 1
        yield current_sample, total_accepts / (num_iter + 1)


def hamiltonian_chain(num_iters, dim, distro):
    num_leapfrog_steps = 20
    epsilon = 0.0005

    inv_cov = inv(distro.cov)

    def grad_energy(x):
        return inv_cov.dot(x - distro.mean)

    # Start with some initial vector
    current_sample = np.zeros((dim,))

    # Also track the acceptance rate
    total_accepts = 0

    # Initialize the energy and the hamiltonian
    energy = -1 * distro.logpdf(current_sample)
    gradient = grad_energy(current_sample)

    for num_iter in range(num_iters):
        # For each iteration, we start with a random momentum
        M_sqrt = 0.1
        momentum = M_sqrt * np.random.randn(dim)

        # Pseudo: hamiltonian = kinetic_energy + potential energy
        hamiltonian = np.sum(np.square(momentum)) / 2 + energy

        # Initialize running variables
        proposed_sample = current_sample
        gradient_new = gradient

        # In Leapfrog integration, we need to make 2*L half steps. (L => num_leapfrog_steps)
        # Therefore, we
        # -- Start with a half step
        # -- Make L full steps
        # -- Correct for a half step
        momentum -= epsilon * gradient_new / 2
        for t in range(num_leapfrog_steps):
            proposed_sample += epsilon * momentum / M_sqrt**2
            gradient_new = grad_energy(proposed_sample)
            momentum -= epsilon * gradient_new

        momentum += epsilon * gradient_new / 2  # Correct the final half step

        # Calculate the new hamiltonian after the momentum-accelerated steps
        energy_new = -1 * distro.logpdf(proposed_sample)
        hamiltonian_new = np.sum(np.square(momentum)) / 2 + energy_new

        # Now the acceptance depends on the difference in Hamiltonian before and after
        delta_hamiltonian = hamiltonian_new - hamiltonian

        if delta_hamiltonian < 0:
            accept = True
        elif random.random() < np.exp(-1 * delta_hamiltonian):
            accept = True
        else:
            accept = False
        total_accepts += int(accept)

        if accept:
            # If we accept, then also carry over the gradient and energy
            gradient = gradient_new
            current_sample = proposed_sample
            energy = energy_new

        yield current_sample, total_accepts / (num_iter + 1)
