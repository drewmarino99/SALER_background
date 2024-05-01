from behrens_buhring_P_star import dGamma_dEf, monte_carlo
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import corner


# This one only for 11C
m_e = 511  # keV
rho = 0.75442
abv = (1 - 1 / 3 * rho ** 2) / (1 + rho ** 2)

mass = 11
charge = 6
sensitivityRatio = -1.2
m_f = mass * 1822.89
delta = (1982.4 - m_e) / 511
m_i = m_f + delta
q_max = np.sqrt(delta ** 2 - 1)
qs = np.linspace(0, q_max, 10000)
lil_b = 0
real_ys = dGamma_dEf(delta, charge, m_f, qs, abv, lil_b)
real_max = np.amax(real_ys)
print(abv)
print(q_max ** 2 / (2 * m_f) * 511000)


# define uninformative (flat) prior with limits on a and b
def log_prior(theta):
    test_a, test_b, test_scale = theta
    if -1 <= test_a <= 1 and -1 <= test_b <= 1 and test_scale > 0:
        return 0.0
    return -np.inf


# define log-likelihood function.  I have no idea if this is correct
def log_likelihood(theta, x, y, yerr):
    test_a, test_b, test_scale = theta
    model = test_scale * dGamma_dEf(delta, charge, m_f, x, test_a, test_b)
    sigma2 = yerr ** 2  # + model ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2)  # + np.log(sigma2))


# log probability: posterior
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


# Sample from CDF to generate data to fit to
qs, monte_carlo_data = monte_carlo(delta, charge, m_f, qs, abv, lil_b, int(1e7))
monte_carlo_error = np.sqrt(monte_carlo_data)
monte_carlo_error[monte_carlo_error == 0] += 1

# negative log likelihood defined as in https://emcee.readthedocs.io/en/stable/tutorials/line/
# Lambda seems a weird choice but it's what they did
nll = lambda *args: -log_likelihood(*args)

# Initial guesses at parameters (using the real values for now)
# scale_guess = np.amax(monte_carlo_data) / real_max
scale_guess = 1
initial = np.array([abv, lil_b, scale_guess])

x = qs
y = monte_carlo_data
yerr = monte_carlo_error

plt.plot(qs, np.divide(monte_carlo_data, scale_guess))
plt.plot(qs, real_ys)
plt.show()

# max log likelihood to get initial guesses for MCMC
soln = minimize(nll, initial, args=(x, y, yerr))
a_ml, b_ml, scale_ml = soln.x

print(abv, lil_b, scale_guess)
print(a_ml, b_ml, scale_ml)
# Note scale should be ~1: everything is normalized!

plt.plot(qs, np.divide(monte_carlo_data, scale_guess), label='MC Data')
plt.plot(qs, dGamma_dEf(delta, charge, m_f, qs, a_ml, b_ml), label='dGamma/dEf')
plt.legend()
plt.show()

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
sampler.run_mcmc(pos, 50000, progress=True)
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["a", "b", "scale"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
fig = corner.corner(
    flat_samples, labels=labels, truths=[abv, lil_b, 1/scale_guess]
)
print(np.mean(flat_samples, axis=0))
plt.show()

# Go back to minuit/curve_fit/whatever for now and try nonlinearity, resolution, etc.
# Take BeEST laser centroids and scatter each in a gaussian centered at measured value with width uncertainty on centroid
# Use this to apply to recoil data and fit to little a, this will give you the remaining uncertainty on the calibration applied to little a
# constant noise, fano noise, inhomogeneity noise