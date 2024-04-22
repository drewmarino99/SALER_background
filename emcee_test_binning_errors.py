# Testing for binning errors
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner


# define uninformative (flat) prior with limits on a and b
def log_prior(theta):
    test_a, test_b, test_c = theta
    if -100 <= test_a <= 100 and -1 <= test_b and -1 <= test_c:
        return 0.0
    return -np.inf


# define log-likelihood function
def log_likelihood(theta, x, y, yerr):
    test_a, test_b, test_c = theta
    model = test_a * x ** 2 + test_b * x + test_c
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2)  # + np.log(sigma2))


# log probability: posterior
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


def monte_carlo(points, bincount):
    pdf_x = np.linspace(0, endpoint, num=bincount)
    binwidth = pdf_x[1] - pdf_x[0]
    pdf = truth(pdf_x)
    pdf = pdf / (np.sum(pdf) * binwidth)
    # print(binwidth, np.sum(pdf))
    cdf = np.zeros_like(pdf)
    for ix in range(len(pdf) - 1):
        cdf[ix + 1] = cdf[ix] + (pdf_x[ix + 1] - pdf_x[ix]) * 0.5 * (pdf[ix] + pdf[ix + 1])
        # take ratio with analytical via integration
        # possibly need cdf one longer
        # Trapezoid or simpson integration?
        # Check what CDF is before doing next line: should be one
    cdf = cdf / cdf[-1]
    # print(cdf)
    # plt.plot(pdf_x, (integral_fn(pdf_x) - cdf))
    # plt.show()
    y_samples = np.random.uniform(0, 1, points)
    x_samples = np.interp(y_samples, cdf, pdf_x)
    counts, bins = np.histogram(x_samples, bins=bincount, density=True)

    return bins[1:] - (bins[1] / 2 - bins[0] / 2), counts


numtrials = 100

labels = ["a", "b", "c"]

errors = np.zeros((numtrials, 3))
for i in range(numtrials):
    print(i)
    b = np.random.uniform()
    a = np.random.uniform(4,10)
    truth = np.poly1d([a, b, 0])
    c = -np.min(truth(np.linspace(0, 100, 100)))

    integral_fn = np.poly1d([a / 3, b / 2, c, 0])
    endpoint = 1
    integral = integral_fn(endpoint) - integral_fn(0)
    a /= integral
    b /= integral
    c /= integral
    integral_fn = np.poly1d([a / 3, b / 2, c, 0])
    truth = np.poly1d([a, b, c])
    # print(truth)

    x, y = monte_carlo(1000000, 1000)
    # plt.plot(x, y)
    # plt.plot(x, truth(x))

    yerr = np.sqrt(y)  # counting statistics
    yerr[yerr == 0] = 1

    # Probably don't need to do emcee here yet either:
    # the difference of PDF vs analytical should be sqrt(N) independent of bin size

    initial = np.array([0, 0.5, 5])  # initial guess in middle of range
    nll = lambda *args: -log_likelihood(*args)
    soln = minimize(nll, initial, args=(x, y, yerr))
    a_ml, b_ml, c_ml = soln.x
    print(soln.x, [a, b, c])

    pos = (soln.x) + (1e-2 * np.random.randn(32, 3))
    # print(pos.shape)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, 5000, progress=True)
    # fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)

    # axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    # print(flat_samples.shape)
    # fig = corner.corner(
    #     flat_samples, labels=labels, truth=[a, b, c], show_titles=True
    # )
    a_mc, b_mc, c_mc = np.mean(flat_samples, axis=0)
    # plt.plot(x, y)
    # plt.plot(np.linspace(0, endpoint, 100), np.poly1d([a_mc, b_mc, c_mc])(np.linspace(0,endpoint,100)))
    # plt.plot(np.linspace(0, endpoint, 100), truth(np.linspace(0, endpoint, 100)))
    errors[i] = [100*(a - a_mc)/a_mc, 100*(b - b_mc)/b_mc, 100*(c - c_mc)/c_mc]
    # plt.show()
for i in range(3):
    plt.figure()
    plt.hist(errors[:, i], label=labels[i])
    plt.xlabel('Truth-measured (%)')
    plt.ylabel('Counts')
    plt.legend()
plt.show()

# Go back to where I left off, however:
# * Fitting procedure has a bias even without introducing one
# * Look at relative difference.  Set initial offset to zero then introduce new offset and see what the change in the fit is
# * In parallel Leendert will look at it and try to help
# * First steps: offset, linearity, quadricity, resolution, etc.