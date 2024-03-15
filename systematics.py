import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from iminuit import Minuit
from iminuit.cost import LeastSquares
from behrens_buhring_P_star import dGamma_dEf

# This one only for 11C
m_e = 511  # keV
rho = 0.75442
abv = (1 - 1 / 3 * rho ** 2) / (1 + rho ** 2)
print(abv)
mass = 11
charge = 6
sensitivityRatio = -1.2
m_f = mass * 1822.89
delta = (1982.4 - m_e) / 511
m_i = m_f + delta
q_max = np.sqrt(delta ** 2 - 1)
qs = np.linspace(0, q_max, 1000)
lil_b = 0
real_max = np.amax(dGamma_dEf(delta, charge, m_f, qs, abv, lil_b))
print(q_max**2/(2 * m_f)*511000)


# noinspection DuplicatedCode
def monte_carlo_abv(points, blur_ev):
    y_cdf = dGamma_dEf(delta, charge, m_f, qs, abv, lil_b,)
    for i, el in enumerate(y_cdf):
        if i == 0:
            pass
        else:
            y_cdf[i] = y_cdf[i - 1] + el

    y_cdf = y_cdf / y_cdf[-1]
    x_res = np.interp(np.random.rand(points), y_cdf, qs)

    momentum_blur = np.sqrt(500 * blur_ev / 511000)
    x_res_blurred = np.random.normal(x_res, scale=momentum_blur)

    const = 0.07
    lin = 1.005
    quad = 0.0005
    polynom_nonlinearity = np.polynomial.polynomial.Polynomial([const, lin, quad])
    x_res_blurred = polynom_nonlinearity(x_res_blurred)

    # plt.hist(x_res, histtype='step', bins=100, label='unblurred', range=[0, 3])
    # plt.hist(x_res_blurred, histtype='step', bins=100, label='blurred', range=[0, 3])
    # plt.legend()
    # plt.show()

    counts, bins = np.histogram(x_res_blurred, bins=1000, range=[qs[0], qs[-1]])
    xs, ys = bins[1:], counts

    ys_err = np.sqrt(ys)
    ys_err[ys_err == 0] = 1
    scale_guess = np.amax(ys) / real_max

    # Without b
    least_squares = LeastSquares(xs, ys, ys_err,
                                 lambda x, a, scale: scale * dGamma_dEf(delta, charge * 0,
                                                                        m_f, x, a, 0))

    def offsetdGamma(x, a, scale, offset):
        expandedXrange = np.arange(x[0] - np.abs(offset), x[-1] + np.abs(offset), x[1] - x[0])
        y = scale * dGamma_dEf(delta, charge * 0, m_f, expandedXrange, a, 0)
        return np.interp(x + offset, expandedXrange, y)

    least_squares = LeastSquares(xs, ys, ys_err, offsetdGamma)

    m = Minuit(least_squares, a=0.516, scale=scale_guess, offset=0.07)

    m.migrad()
    m.hesse()
    del xs, ys, counts, bins
    return [m.values['a'], m.errors['a'], m.values['offset']]


# res = monte_carlo_abv(1000000, 1.4)
res = Parallel(n_jobs=12, verbose=1)(delayed(monte_carlo_abv)(1000000, 1.4) for _ in range(1000))
vals = list(zip(*res))[0]
errors = list(zip(*res))[1]
offsets = list(zip(*res))[2]
plt.hist(vals, bins=20)
plt.axvline(np.mean(vals), color='r')
for ii in [-3, -2, -1, 1, 2, 3]:
    plt.axvline(ii * np.mean(errors) + np.mean(vals), color='k')
print(np.mean(vals), np.mean(errors))
plt.show()
plt.hist(offsets)
plt.show()
