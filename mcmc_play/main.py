import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, wishart
from mcmc_play.util.util import plot_cov_ellipse, compute_KL
from mcmc_play.util.util_proposals import random_walk_chain, gibbs_chain, hamiltonian_chain
from itertools import islice
from scipy.stats import chi2
colors = ['b', 'r', 'c', 'g', 'm', 'y', 'k', 'w']


# Make the distribution that we want to sample from
dim = 30
distro_mean = np.random.randn(dim)
df = 30  # Degrees of freedom for the Wishart distribution
distro_cov = wishart(df, np.eye(dim)/df).rvs()
distro_marginal_var = np.diag(distro_cov)

print('The 1 sigma lengths along the principal axes')
print(np.sort(np.sqrt(np.linalg.eigvals(distro_cov) * chi2.ppf(0.68, dim))))

distro = multivariate_normal(mean=distro_mean, cov=distro_cov)

num_iter = 10000

# MCMC parameters
num_steps_burnin = 50
num_steps_subsample = 20
num_samples_total = int((num_iter - num_steps_burnin) / num_steps_subsample)


# Some matplotlib magic
fig, axarr = plt.subplots(4, 1)
axarr[0].set_xlim([-3., 3.])
axarr[0].set_ylim([-3., 3.])
axarr[1].set_xlim([0, num_samples_total])
axarr[1].set_ylim([0.0, 3.0])
axarr[1].set_title('KL divergences per axis')
axarr[2].set_xlim([0, num_samples_total])
axarr[2].set_ylim([0.0, 1.0])
axarr[2].set_title('Acceptance rate')
axarr[3].set_xlim([0, num_samples_total])
axarr[3].set_ylim([0.0, 3.0])
axarr[3].set_title('Average kl divergence')

moments = np.zeros((dim*2))

plot_cov_ellipse(distro_cov[:2, :2], distro_mean[:2], ax=axarr[0], volume=0.5)

total_accepts = 0
# Use an islice here to discard the samples during burn in and subsample every nth step of the chain
for num_sample, (sample, acceptance) in enumerate(islice(hamiltonian_chain(num_iter, dim, distro),
                                                         num_steps_burnin,
                                                         None,
                                                         num_steps_subsample)):
    moments[:dim] += sample
    moments[dim:2*dim] += np.square(sample)

    axarr[0].scatter(sample[0], sample[1])

    if num_sample % 5 == 0 and num_sample > 1:
        # Calculate the emperical moments (note that in reality, you wouldn't know the true moment)
        sample_mean = moments[:dim] / (num_sample + 1)
        sample_second_moment = moments[dim:2*dim] / (num_sample + 1)
        sample_var = sample_second_moment - np.square(sample_mean)

        # Compute the KL divergence
        kl_divergence = compute_KL(distro_mean, distro_marginal_var, sample_mean, sample_var)

        # Pyplot magic
        axarr[0].set_title(f'KL div: {np.mean(kl_divergence):8.3f}')

        for i, kld in enumerate(kl_divergence):
            axarr[1].scatter(num_sample, kld, c=(colors[i % len(colors)]), label=f'kl{i}', s=5)
        # axarr[1].legend()

        # PLot the acceptance rate
        axarr[2].scatter(num_sample, acceptance, c='r')

        # Plot the average kl divergence
        axarr[3].scatter(num_sample, np.mean(kl_divergence), c='r')
        plt.savefig(f'im/{num_sample}.png')

        if num_sample % 50 == 0:
            print(f'{num_sample:6.0f} - acceptance {acceptance:.3f} - mean kl {np.mean(kl_divergence):8.3f}')
