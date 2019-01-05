# Comparing three popular methods for MCMC
In this post, we will compare three popular Markov Chain Monte Carlo methods (MCMC). MCMC is a method garanteed to sample from any distribution, given that one satisfies some conditions. One might ask for the use of this sampling. We know how to exactly sample the Gaussian, Poisson, Exponential, Multinomial, Bernoulli and so on, don't we? We usually use MCMC for way more complicated distributions, such as posteriors for hierarchical models, neural network posteriors, or posteriors of general probabilistic graphical models.

## Comparison to variational inference
By sampling from the distribution, MCMC falls into the category of __approximate inference__. In approximate inference, we are to __approximate__ some property of a distribution. Usually, because exact inference is too hard. Another big player in the approximate inference toolbox is variational inference (VI). In VI, we approximate the complicated distribution with a simpler family of distributions. The best distinction between VI and MCMC relates to the bias-variance trade off. I associate this explanation mostly with Max Welling. 

  * VI: *high* bias, *low* variance. The variational approximation might be way different from the original distribution. Moreover, if the approximation itself is framed as an optimization procedure, then it might get stuck in a local optimum. That means that properties we estimate from the approximate distribution are biased from the estimates of the original. However, the variational approximation itself is a deterministic procedure. Therefore, it has low variance.

  * MCMC: *low* bias, *high* variance. In the infinite limit, the Markov Chain converges to walking along the distribution. So, if the necessary conditions are met, MCMC is an unbiased procedure: any properties we estimate using the samples are unbiased from the properties we would have estimated from the original distribution itself. Unfortunately, this convergence is only guaranteed in the infinite limit. Within finite time, the samples might be way off from the distribution. Moreover, two consequetive jumps in the Markov Chain are highly correlated. For these two reasons, MCMC has high variance.

TL;DR; if you are sure about your model and distribution, use MCMC. If your model is just a wild guess to make predictions, use VI.

# MCMC: proposals and acceptances
The Markov Chain makes steps according to a proposal distribution. At any point <img alt="$x_0$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/e714a3139958da04b41e3e607a544455.svg" align="middle" width="15.888015000000001pt" height="14.102549999999994pt"/>, the chain will jump to point <img alt="$x_1$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/277fbbae7d4bc65b6aa601ea481bebcc.svg" align="middle" width="15.888015000000001pt" height="14.102549999999994pt"/> with probability: <img alt="$q(x_1|x_0)$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/0335ffb642ef7828d6e8aaaca6de6e5f.svg" align="middle" width="58.65617999999999pt" height="24.56552999999997pt"/>. This <img alt="$q$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg" align="middle" width="7.898533500000002pt" height="14.102549999999994pt"/> must satisfy an equation called __detailed balance__. If <img alt="$q$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg" align="middle" width="7.898533500000002pt" height="14.102549999999994pt"/> satisfies detailed balance, then we are garanteed to sample from the right distribution in the infinite limit. 

Detailed balance reads as follows:

<img alt="$p(x_0) \mathcal{T}(x_0 \rightarrow x_1) = p(x_1) \mathcal{T}(x_1 \rightarrow x_0)$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/2d2a1c18280365da0952b64b25ee9e93.svg" align="middle" width="266.91934499999996pt" height="24.56552999999997pt"/>

Where we use a different symbol for the transition probability, <img alt="$\mathcal{T}$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/6937e14ec122765a9d014f2cbcf4fcfe.svg" align="middle" width="13.08219pt" height="22.381919999999983pt"/>. The trick to alway satisfy this equation is to decompose the transition into a proposal and acceptance step:

<img alt="$p(x_0) q(x_1|x_0)A[x_0 \rightarrow x_1] = p(x_1) q(x_0|x_1)A[x_1 \rightarrow x_0]$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d65c35c045a9ee9b3bb52ec0fabd06fa.svg" align="middle" width="375.32434499999994pt" height="24.56552999999997pt"/>

Now we can use any proposal and use it with acceptance probability:

<img alt="$A[x_0 \rightarrow x_1] = min[1, \frac{p(x_1)q(x_0|x_1)}{p(x_0)q(x_1|x_0)} ]$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/b849447d36cfecc093d410515153acf9.svg" align="middle" width="239.87749499999995pt" height="33.14091000000001pt"/>

# Comparing proposal distributions

So really all the design choices for MCMC are in the proposal distributions. In this project, we will compare three different proposal distributions (abbreviated to <img alt="$q$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg" align="middle" width="7.898533500000002pt" height="14.102549999999994pt"/>).

  * Conditional Gaussian: from a point <img alt="$x_0$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/e714a3139958da04b41e3e607a544455.svg" align="middle" width="15.888015000000001pt" height="14.102549999999994pt"/>, we sample <img alt="$epsilon$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/3d97b54f274c7bdc4d442066f97af7af.svg" align="middle" width="52.162275pt" height="22.745910000000016pt"/> from a unit Gaussian and propose <img alt="$x_1 = x_0 + \sigma \epsilon$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/36cc683186e891eb1d219157770c8ed1.svg" align="middle" width="91.97726999999999pt" height="19.10667000000001pt"/>. <img alt="$q(x_1| x_0) = \mathcal{N}(x_0, \sigma^2)$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d451f992a7436a3e72e6bec1a3ebbe7c.svg" align="middle" width="150.468615pt" height="26.70657pt"/>
  * Gibbs sampling: from a point <img alt="$x_0$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/e714a3139958da04b41e3e607a544455.svg" align="middle" width="15.888015000000001pt" height="14.102549999999994pt"/>, we sample <img alt="$x_{0i}$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/d8baacbc2c2972c7dad69d338b91881b.svg" align="middle" width="20.521545pt" height="14.102549999999994pt"/> conditioned on all indices in <img alt="$x_0$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/e714a3139958da04b41e3e607a544455.svg" align="middle" width="15.888015000000001pt" height="14.102549999999994pt"/> except i. We denote this <img alt="$x_{0-i}$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/ecc9a9478242d939e03d6c56be199ced.svg" align="middle" width="30.757155pt" height="14.102549999999994pt"/>. <img alt="$q(x_1|x_0) = p(x_{1i}|x_{0-i})$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/38428807c0c9769b4779b60a2ba18ccd.svg" align="middle" width="159.09679500000001pt" height="24.56552999999997pt"/>
  * Hamiltonian Monte Carlo: from a point, we roll a hypothetical ball along the probability surface. We propose a sample where the ball lands after <img alt="$T$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg" align="middle" width="11.845020000000003pt" height="22.381919999999983pt"/> seconds. More explanation [here](link_to_betancourt_paper)

# Experiment
We will compare these proposal distributions on a high-dimensional Gaussian distribution. In high dimensions, the probability mass of a Gaussian distribution concentrates on a thin shell. Therefore, it is considered hard to traverse the entire distribution to obtain unbiased samples.

I want to add that each proposal distribution has its advantages and disadvantages. The results on a single Gaussian distribution might not transfer to another distribution.

# Results
We need a metric to see how well the samples cover the entire distribution. Doing MCMC in the wild, this metric is very hard to find. We cannot evaluate the distribution, so how could we measure if samples covered it? However, we have the fortunate position that we CAN evaluate our distribution. Simply for the reason that this is a toy problem. As such, we will compare how fast the KL divergence between the _true_ distribution and emperical distribution goes to zero.

![image_spherical_gaussian](?raw=true)

![image_gibbs](?raw=true)

![image_hmc](?raw=true)

# Discussion

We make the following observations

 * The KL divergence goes down slowest for the spherical Gaussian. Gibbs sampling is faster. For any axis with length scale <img alt="$L$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.145420000000001pt" height="22.381919999999983pt"/>, the Spherical Gaussian is <img alt="$\frac{L}{\sigma}$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/175949f9de8aa73a6efd3e4a5c35e81e.svg" align="middle" width="9.018405pt" height="28.61198999999999pt"/> lenghts away. Therefore, it needs <img alt="$(\frac{L}{\sigma})^2$" src="https://github.com/RobRomijnders/mcmc_proposals/blob/master/svgs/5567a67f36c4bf22ee59520ce6d73553.svg" align="middle" width="32.229285pt" height="28.61198999999999pt"/> steps to traverse this axis. However, Gibbs sampling samples from this entire length at each step. Therefore, Gibbs sampling can traverse the distribution faster.
 * Hamiltonian Monte Carlo is even faster than Gibbs sampling. There's two ways to look at this
   * HMC uses the gradient of the probability surface. Analogous to optimization holds the addage "any optimization using the gradient is least as good as gradient-free optimization". Like so in desigining a Markov Chain. The gradient points to a higher probability density. We might as well use that information
   * HMC is designed to follow the _typical set_ of the distribution. The typical set of a distribution is where most of the probability mass is concentrated ([here](link-to-wikipedia) is a nicer description). Once the Chain finds the typical set, it can rapidly traverse it.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com
