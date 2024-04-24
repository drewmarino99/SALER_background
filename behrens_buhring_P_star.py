import matplotlib.pyplot as plt
import numpy as np
import sympy
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.colors as clr
import uncertainties
from uncertainties import unumpy
from uncertainties.umath import *
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter


# Function from page 169 of Behrens & Buhring 1982 (Electron radial wave functions and nuclear beta decay)
# Figure 5.2
# params:
#   x: |q|/|q_max|, where q is momentum transfer to the nucleus and |q_max| is sqrt(d^2-m_e^2)
#       range: 0 to 1, obviously
#   d: Delta. Mass difference between initial and final atom
#   m_elec: electron mass in whatever units you're using
def p_star(x, d, m_elec):
    a = d ** 2 / (d ** 2 - m_elec ** 2)
    b = (4 * d ** 2 - m_elec ** 2) / (d ** 2 - m_elec ** 2)
    c = 3 * d ** 2 * (d ** 2 + m_elec ** 2) / ((d ** 2 - m_elec ** 2) ** 2)
    return x ** 2 * ((1 - x ** 2) ** 2 / ((a - x ** 2) ** 3)) * (x ** 4 - b * x ** 2 + c)


# First order expansion of fermi function.
# Z is atomic number of daughter, taken to be negative for beta plus decay.
# W_e is beta energy in units of energy/m_e
def fermi(Z, W_e):
    return 1 + Z * W_e / np.sqrt(W_e ** 2 - 1)


# Integral from SALER Prep email chain.  m, n, and l are the powers in the integral
# E_0 is beta spectrum endpoint units of electron rest mass
def i_mnl(E_0, m, n, l, bounds):
    e1s = bounds[0]
    e2s = bounds[1]
    # TODO: replace this implementation with a dict lookup in case we add more terms to this
    # Preevaluated integrals for code speedup
    if (m, n, l) == (1, 1, 0):
        def integral(e_e, e_o):
            return -e_e ** 3 / 3 + e_e ** 2 * e_o / 2

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (0, 1, 0):
        def integral(e_e, e_o):
            return -e_e ** 2 / 2 + e_e * e_o

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (0, 0, 0):
        def integral(e_e, e_o):
            return e_e

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (0, 0, 2):
        def integral(e_e, e_o):
            return e_e ** 3 / 3 - e_e

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (0, 2, 0):
        def integral(e_e, e_o):
            return e_e ** 3 / 3 - e_e ** 2 * e_o + e_e * e_o ** 2

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (2, 1, -1):
        def integral(e_e, e_o):
            return -e_e ** 2 * np.sqrt(e_e ** 2 - 1) / 3 + e_e * e_o * np.sqrt(e_e ** 2 - 1) / 2 \
                   + e_o * np.log(e_e + np.sqrt(e_e ** 2 - 1)) / 2 - 2 * np.sqrt(e_e ** 2 - 1) / 3

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (1, 0, -1):
        def integral(e_e, e_o):
            return np.sqrt(e_e ** 2 - 1)

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (1, 0, 1):
        def integral(e_e, e_o):
            return e_e ** 2 * np.sqrt(e_e ** 2 - 1) / 3 - np.sqrt(e_e ** 2 - 1) / 3

        return integral(e2s, E_0) - integral(e1s, E_0)
    elif (m, n, l) == (1, 2, -1):
        def integral(e_e, e_o):
            return e_e ** 2 * np.sqrt(e_e ** 2 - 1) / 3 - e_e * e_o * np.sqrt(e_e ** 2 - 1) + e_o ** 2 * np.sqrt(
                e_e ** 2 - 1) \
                   - e_o * np.log(e_e + np.sqrt(e_e ** 2 - 1)) + 2 * np.sqrt(e_e ** 2 - 1) / 3

        return integral(e2s, E_0) - integral(e1s, E_0)
    # In case we add terms, we would need this bit.
    else:
        print(f'Calculating new integral: {m}, {n}, {l}')
        E_e, E_o = sympy.symbols('E_e E_o')
        res = sympy.integrate(E_e ** m * (E_o - E_e) ** n * sympy.sqrt(E_e ** 2 - 1) ** l, E_e)
        print(res, f'm: {m}, n: {n}, l: {l}')
        output = np.zeros(len(e1s))
        for i in range(len(e1s)):
            output[i] = res.evalf(subs={E_e: e2s[i], E_o: E_0}) - res.evalf(subs={E_e: e1s[i], E_o: E_0})
    # plt.plot(qs ** 2 / (2 * m_f) * 511000, output, label=f'm: {m}, n: {n}, l: {l}')
    return output


# Function from Behrens/Buhring with corrections due to arbitrary terms.  For now, derived via Mathematica
# in little_a_context_behrens_buhring.nb
def p_star_little_a(mi, mf, pf, xi, a, b):
    delta = mf - mi
    prefactor = xi * (1 / (96 * np.pi ** 3 * (pf ** 2 - delta ** 2) ** 3)) * np.sqrt(mf ** 2 + pf ** 2) * (
            pf + pf ** 3 - pf * delta ** 2)
    term1 = pf ** 2 * (
            1 + a + (2 - 12 * b * mi) * pf ** 2 + pf ** 4 + a * pf ** 2 * (-4 + 6 * mi ** 2 + (-6 + pf) * pf))
    term2 = 6 * (b - (1 + a) * mi) * pf ** 2 * (-1 + pf ** 2) * delta
    term3 = (3 + 8 * (-1 + 3 * b * mi) * pf ** 2 + pf ** 4 + a * (
            3 + pf ** 2 * (4 - 12 * mi ** 2 + pf * (12 + pf)))) * delta ** 2
    term4 = 6 * (b - (1 + a) * mi) * (-1 + 2 * pf ** 2) * delta ** 3
    term5 = (-6 + 12 * b * mi - 6 * a * mi ** 2 + 6 * a * pf + 5 * (1 + a) * pf ** 2) * delta ** 4
    term6 = 6 * (b - (1 + a) * mi) * delta ** 5
    term7 = 3 * (1 + a) * mi * delta ** 6
    res = prefactor * (term1 + term2 + term3 - term4 - term5 + term6 + term7)
    return res


# Recoil spectrum from beta decay.
# Parameters: Delta: q-value,  Z: Daughter atomic number (negative if beta plus),
#   a: beta-nu angular correlation coefficient, b: Fierz coefficient
def dGamma_dEf(delta, Z, mf, pf, a, b):
    G = 1e-5 / (0.938 ** 2)  # G_fermi in GeV^-2
    E_0 = delta
    E_f = np.sqrt(pf ** 2 + mf ** 2)
    W_e_upper = ((delta + pf) ** 2 + 1) / (2 * (delta + pf))
    W_e_lower = ((delta - pf) ** 2 + 1) / (2 * (delta - pf))
    e_bounds = [W_e_lower, W_e_upper]
    # plt.plot(qs ** 2 / (2 * m_f) * 511000, W_e_lower, label='W_e lower')
    # plt.plot(qs ** 2 / (2 * m_f) * 511000, W_e_upper, label='W_e upper')
    # plt.xlabel('E_recoil')
    # plt.ylabel('W_e 1&2')
    # plt.legend()
    # plt.show()
    # Equation from Leendert
    res = G ** 2 * E_f / (4 * np.pi ** 3) * (
            i_mnl(E_0, 1, 1, 0, e_bounds)
            + b * i_mnl(E_0, 0, 1, 0, e_bounds)
            + a / 2 * (pf ** 2 * i_mnl(E_0, 0, 0, 0, e_bounds) - i_mnl(E_0, 0, 0, 2, e_bounds) - i_mnl(E_0, 0, 2, 0,
                                                                                                       e_bounds))
            + 1 / 137 * Z * (i_mnl(E_0, 2, 1, -1, e_bounds) + a / 2 * (
            pf ** 2 * i_mnl(E_0, 1, 0, -1, e_bounds) - i_mnl(E_0, 1, 0, 1, e_bounds) - i_mnl(E_0, 1, 2, -1,
                                                                                             e_bounds)))
    )
    return res/(np.sum(res) * (pf[1]-pf[0]))  # normalize


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def monte_carlo(Delta, Z_daught, m_f, qs, lil_a, lil_b, points):
    bincount = len(qs)
    pdf = dGamma_dEf(Delta, Z_daught, m_f, qs, lil_a, lil_b)
    pdf_x = qs
    cdf = np.zeros_like(pdf)

    # NEW METHOD FROM POLYNOMIAL TEST
    for ix in range(len(pdf) - 1):
        cdf[ix + 1] = cdf[ix] + (pdf_x[ix + 1] - pdf_x[ix]) * 0.5 * (pdf[ix] + pdf[ix + 1])
    cdf = cdf / cdf[-1]
    # print(cdf)
    # plt.plot(pdf_x, (integral_fn(pdf_x) - cdf))
    # plt.show()
    y_samples = np.random.uniform(0, 1, points)
    x_samples = np.interp(y_samples, cdf, pdf_x)
    counts, bins = np.histogram(x_samples, bins=bincount, density=True)
    return bins[1:] - (bins[1] / 2 - bins[0] / 2), counts

    # OLD METHOD
    # y_cdf = pdf
    # for i, el in enumerate(y_cdf):
    #     if i == 0:
    #         pass
    #     else:
    #         y_cdf[i] = y_cdf[i - 1] + el
    #
    # y_cdf = y_cdf / y_cdf[-1]
    # x_res = np.interp(np.random.rand(points), y_cdf, qs)
    #
    # counts, bins = np.histogram(x_res, bins=1000000, range=[qs[0], qs[-1]])
    # # functional_form = dGamma_dEf(Delta, Z_daught, m_f, qs, lil_a, lil_b)
    # # functional_max = np.max(functional_form)
    # # avg_bin_max = np.max(moving_average(counts, 10))
    # # plt.plot(bins[:-1]/bins[-1], counts / avg_bin_max, drawstyle='steps-mid', label=f'MC, Delta = {Delta}, points = {points}')
    # # plt.plot(qs/q_max, functional_form / functional_max, label=f'Delta = {Delta}')
    # return bins[:-1] + 0.5 * (bins[1]-bins[0]), counts


if __name__ == '__main__':
    Z_daught = 0
    N_daught = 7
    m_f = (Z_daught + N_daught) * 1822.89
    Delta = 3
    m_i = m_f + Delta
    lil_a = 1
    lil_b = 0
    q_max = np.sqrt(Delta ** 2 - 1)
    qs = np.linspace(0, q_max, 1000)
    # for a in [-1, -1/3, 1/3, 1]:
    #     xs, ys = monte_carlo(Delta, Z_daught, m_f, qs, a, lil_b, int(1e6))
    #     ys2 = dGamma_dEf(Delta, Z_daught, m_f, qs, a, lil_b)
    #     # plt.plot(xs**2/(2*m_f) * 511000, ys * ys2.max()/ys.max(), drawstyle='steps-mid', label=f'a = {a:.2f}')
    #     plt.plot(qs**2/(2*m_f) * 511000, ys2, label=f'a = {a:.2f}')
    #     pass
    # plt.legend()
    # plt.suptitle('Effect of $a_{βν}$ on spectral shape')
    # plt.title('Mass = 3 Da, Charge = 0, Beta endpoint = 3 m$_e$', fontsize=10)
    # plt.xlabel('Nuclear recoil energy [eV]')
    # plt.ylabel('Spectral density')
    # plt.yticks([])
    # plt.show()
    # exit()

    # result2 = p_star_little_a(m_i, m_f, qs, 1, lil_a, lil_b)
    # plt.plot(qs**2/(2*m_f) * 511000, result2)
    # plt.show()

    # points = np.power(10, np.linspace(3, 8, 15)).astype(int)
    # tests = [(delta, lil_a, num_points)
    #          for delta in np.linspace(1.1, 5, 5)
    #          for lil_a in [-1, -1 / 3, 1 / 3, 1]
    #          for num_points in points]
    # # tests = [(1.5, a, 1000000) for a in (-1, -1/3, 1/3, 1)]
    #
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].set_title('a = -1')
    # axs[0, 1].set_title('a = -1/3')
    # axs[1, 0].set_title('a = 1/3')
    # axs[1, 1].set_title('a = 1')
    #
    # errors = np.asarray([(0, 0, 0, 0) for _ in range(len(tests))], dtype=np.single)
    # for i, test in enumerate(tests):
    #     print(test)
    #     delta, lil_a, num_points = test
    #     q_max = np.sqrt(delta ** 2 - 1)
    #     qs = np.linspace(0, q_max, 1000)
    #     real_max = np.amax(dGamma_dEf(delta, Z_daught, m_f, qs, lil_a, lil_b))
    #     xs, ys = monte_carlo(delta, Z_daught, m_f, qs, lil_a, lil_b, num_points)
    #     ys_err = np.sqrt(ys)
    #     ys_err[ys_err == 0] = 1
    #     least_squares = LeastSquares(xs, ys, ys_err,
    #                                  lambda x, a, scale: scale * dGamma_dEf(delta, Z_daught, m_f, x, a, lil_b))
    #     scale_guess = np.amax(ys) / real_max
    #     m = Minuit(least_squares, a=lil_a, scale=scale_guess)
    #     m.migrad()
    #     m.minos()
    #     m.hesse()
    #     errors[i] = (lil_a, delta, num_points, m.errors['a'])
    # # np.save(r'C:\Users\drewm\OneDrive\Documents\CSM\Research\SALER_background\out.npy', errors)
    # # # for a in [-1, -1/3, 1/3, 1]:
    # # #     curr = errors[np.abs(errors[:, 0] - a) < 0.1].transpose()
    # # #     if np.abs(-1 - a) < 0.1:
    # # #         c = axs[0, 0].scatter(curr[2], curr[1], c=curr[3], norm=clr.LogNorm())
    # # #         fig.colorbar(c, ax=axs[0, 0], label='δa')
    # # #     elif np.abs(-1/3 - a) < 0.1:
    # # #         c = axs[0, 1].scatter(curr[2], curr[1], c=curr[3], norm=clr.LogNorm())
    # # #         fig.colorbar(c, ax=axs[0, 1], label='δa')
    # # #     elif np.abs(1/3 - a) < 0.1:
    # # #         c = axs[1, 0].scatter(curr[2], curr[1], c=curr[3], norm=clr.LogNorm())
    # # #         fig.colorbar(c, ax=axs[1, 0], label='δa')
    # # #     elif np.abs(1 - a) < 0.1:
    # # #         c = axs[1, 1].scatter(curr[2], curr[1], c=curr[3], norm=clr.LogNorm())
    # # #         fig.colorbar(c, ax=axs[1, 1], label='δa')
    # for a in [-1, -1 / 3, 1 / 3, 1]:
    #     curr = errors[np.abs(errors[:, 0] - a) < 0.1].transpose()[1:]
    #     if np.abs(-1 - a) < 0.1:
    #         for delta in np.unique(curr[0]):
    #             temp = curr.transpose()[curr.transpose()[:, 0] == delta].transpose()
    #             axs[0, 0].plot(temp[1], temp[2], label=f'Δ={delta * 0.511:0.1f} MeV')
    #     elif np.abs(-1 / 3 - a) < 0.1:
    #         for delta in np.unique(curr[0]):
    #             temp = curr.transpose()[curr.transpose()[:, 0] == delta].transpose()
    #             axs[0, 1].plot(temp[1], temp[2], label=f'Δ={delta * 0.511:0.1f} MeV')
    #     elif np.abs(1 / 3 - a) < 0.1:
    #         for delta in np.unique(curr[0]):
    #             temp = curr.transpose()[curr.transpose()[:, 0] == delta].transpose()
    #             axs[1, 0].plot(temp[1], temp[2], label=f'Δ={delta * 0.511:0.1f} MeV')
    #     elif np.abs(1 - a) < 0.1:
    #         for delta in np.unique(curr[0]):
    #             temp = curr.transpose()[curr.transpose()[:, 0] == delta].transpose()
    #             axs[1, 1].plot(temp[1], temp[2], label=f'Δ={delta * 0.511:0.1f} MeV')
    # for row in axs:
    #     for ax in row:
    #         ax.plot(points, 10 / np.sqrt(points), 'g-.', label='10/sqrt(N)')
    #         ax.set_xlabel('Number of events')
    #         ax.set_ylabel('δa')
    #         ax.legend()
    #         ax.loglog()
    # plt.tight_layout()
    # plt.show()

    # # Specific applications to mirror nuclei
    # # See Table 1 from 2009.11364
    # # In order: n, 3H, 11C, 13N, 15O, 17F, 19Ne
    # #           b-, b-, b+,  b+,  b+,  b+,  b+
    # # Behrens/Buhring pg 159: delta - m_e = mf-mi for b-, for b+ it's delta + m_e
    m_e = 511  # keV
    rhos = [-2.21086, -2.1053, 0.75442, 0.5596, -0.6302, -1.2955, 1.60203]
    abvs = [(1 - 1 / 3 * r ** 2) / (1 + r ** 2) for r in rhos]
    masses = [1, 3, 11, 13, 15, 17, 19]
    charges = [0, 1, 6, 7, 8, 8, 10]
    sensitivityRatios = [3.6, 4.6, -1.2, -0.7, -0.9, -3.6, -13.1]
    m_fs = [mass * 1822.89 for mass in masses]
    deltas = np.asarray(
        [782.347 + m_e, 18.5906 + m_e, 1982.4 - m_e, 2220.49 - m_e, 2754.0 - m_e, 2760.47 - m_e, 3238.4 - m_e]
    )  # in keV
    deltas = deltas / 511
    m_is = [m_fs[i] + deltas[i] for i in range(len(m_fs))]
    lil_b = 0
    q_maxes = [np.sqrt(delta ** 2 - 1) for delta in deltas]
    qs = np.asarray([np.linspace(0, qmax, 1000000) for qmax in q_maxes])

    # Run each one until sensitivity to dr/r better than 0.1%
    for i in range(len(rhos)):
        sensitivity = 1000
        num_points = int(1e8)
        # Still no Z correction: note * 0
        real_max = np.amax(dGamma_dEf(deltas[i], charges[i], m_fs[i], qs[i], abvs[i], lil_b))
        while sensitivity > 0.1 / 100 and num_points < 1e9:
            xs, ys = monte_carlo(deltas[i], charges[i] * 0, m_fs[i], qs[i], abvs[i], lil_b, num_points)
            ys_err = np.sqrt(ys)
            ys_err[ys_err == 0] = 1
            scale_guess = np.amax(ys) / real_max
            # # With b
            # least_squares = LeastSquares(xs, ys, ys_err,
            #                              lambda x, a, b, scale: scale * dGamma_dEf(deltas[i], charges[i],
            #                                                                        m_fs[i], x, a, b))
            # m = Minuit(least_squares, a=0, scale=scale_guess, b=0)

            # Without b
            least_squares = LeastSquares(xs, ys, ys_err,
                                         lambda x, a, scale: scale * dGamma_dEf(deltas[i], charges[i] * 0,
                                                                                m_fs[i], x, a, 0))
            m = Minuit(least_squares, a=0, scale=scale_guess)

            # # Only b
            # least_squares = LeastSquares(xs, ys, ys_err,
            #                              lambda x, b, scale: scale * dGamma_dEf(deltas[i], charges[i] * 0,
            #                                                                     m_fs[i], x, abvs[i], b))
            # m = Minuit(least_squares, b=0, scale=scale_guess)

            m.migrad()
            m.hesse()

            # Output results
            print(list(['n', '3H', '11C', '13N', '15O', '17F', '19Ne'])[i])
            # Without b
            print(f'Points: {num_points:,},', f'Delta rho / rho: {sensitivity:.5f},', f'Real a: {abvs[i]:.5f},',
                  f"Fit a:{m.values['a']:.5f} +/- {m.errors['a']:0.1e}")

            # # With b
            # print(f'Points: {num_points:,},', f'Delta rho / rho: {sensitivity:.5f},', f'Real a: {abvs[i]:.5f},', f"Fit a:{m.values['a']:.5f} +/- {m.errors['a']:0.1e}, Fit b:{m.values['b']:.5f} +/- {m.errors['b']:0.1e}")

            # # Only b
            # print(f"Points: {num_points:,}, Fit b:{m.values['b']:.5f} +/- {m.errors['b']:0.1e}")

            # print(m.covariance)

            error = m.errors['a']
            sensitivity = np.abs((error / abvs[i]) / sensitivityRatios[i])

            fit = m.values['scale'] * dGamma_dEf(deltas[i], charges[i] * 0,
                                                 m_fs[i], xs, m.values['a'], 0)
            num_points *= 10
            fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
            axs[0].plot(xs ** 2 / (2 * m_fs[i]) * 511000, ys, linewidth=1.1, drawstyle='steps-mid')
            axs[0].plot(xs ** 2 / (2 * m_fs[i]) * 511000, fit, linewidth=0.9, color='r')

            # # Inset zoom for high-statistics plot readability
            # x1, x2 = (xs[np.argmax(fit)] * 0.97, xs[np.argmax(fit)] * 1.03)
            # y1, y2 = fit.max() * 0.97, fit.max() * 1.03
            # inset = axs[0].inset_axes([0.5, 0.1, 0.3, 0.3], xlim=(x1**2/(2*m_fs[i]) * 511000, x2**2/(2*m_fs[i]) * 511000), ylim=(y1, y2), xticklabels=[],
            #                           yticklabels=[])
            # inset.plot(xs**2/(2*m_fs[i]) * 511000, ys, linewidth=1.1, drawstyle='steps-mid')
            # inset.plot(xs**2/(2*m_fs[i]) * 511000, fit, linewidth=0.9, color='r')
            # axs[0].indicate_inset_zoom(inset, edgecolor='k', linewidth=1.2)

            titles = ['Neutron', '3H', '11C', '13N', '15O', '17F', '19Ne']
            axs[0].set_title(
                f"iminuit Fit to Low Statistics (1e5) {titles[i]} Decay\nSM a: {abvs[i]:0.5f}, Fit a: {m.values['a']:0.5f}, δa: {m.errors['a']:.2}")
            axs[1].set_xlabel('Nuclear recoil energy [eV]')
            axs[0].set_ylabel('Counts')
            residuals = (ys - fit) / (np.sqrt(ys) + 1)
            axs[1].plot(xs ** 2 / (2 * m_fs[i]) * 511000, residuals, drawstyle='steps-mid')
            axs[1].axhline(0, color='r', linewidth=0.9)
            axs[1].set_ylabel('Residuals / √(N)')

            axs[0].grid()
            axs[1].grid()
            axs[0].yaxis.set_major_formatter(FuncFormatter('{:,.0f}'.format))
            print(np.amax(np.abs(residuals)))
            axs[1].set_ylim([-(np.abs(residuals)).max() - 0.2, (np.abs(residuals)).max() + 0.2])
            plt.tight_layout()
            plt.show()
            break

    # Check sensitivity to V_ud from 10^8 decays
    # Both uncertainty from just rho and from all factors (see PhysRevC.107.015502 Table VI)
    # Get at V_ud  via SALER proposal (Proposal-263700.pdf) eqn 1.8
    # d_rhos = np.abs(np.asarray([0.00071, 0.01198, 0.00220, 0.00299, 0.00190, 0.00062,
    #                             0.00037]) * rhos)  # d_rho/rho * rho from the above block, same order. b free.
    d_rhos = np.abs(np.asarray([0.00067, 0.01202, 0.00032, 0.00034, 0.00027, 0.00032,
                                0.00036]) * rhos)  # d_rho/rho * rho from the above block, same order. b fixed.
    rhos = unumpy.uarray(rhos, d_rhos)

    # Values in PhysRevC.107.015502 Table VI
    fvts = np.array([1028.25, 1113.0, 3893.4, 4621.3, 4344.3, 2269.5, 1704.34])  # [s]
    fvt_errors = unumpy.uarray(fvts, np.array([0.66, 1.0, 1.4, 4.7, 5.7, 1.7, 0.63]))
    fa_fvs = np.array([1.0000, 1.0003, 0.9992, 0.9980, 0.9964, 1.0020, 1.0011])  # [unitless]
    fa_fv_errors = unumpy.uarray(fa_fvs, np.abs((fa_fvs - 1) * 0.2))
    deltas = np.array([1.4902, 1.767, 1.660, 1.635, 1.555, 1.587, 1.533])  # [%]
    delta_errors = unumpy.uarray(deltas, np.array([0.0002, 0.001, 0.004, 0.006, 0.008, 0.010, 0.012]))
    ft_mirrors = np.array([1043.58, 1130.9, 3916.9, 4681.3, 4402.3, 2291.2, 1721.5])
    ft_mirror_errors = unumpy.uarray(ft_mirrors, np.array([0.67, 1.0, 1.9, 4.9, 5.9, 1.9, 1.0]))
    Gf = 1.1663787 * 10 ** -5  # Gf/(h_bar c)^3 [GeV^-2]     Uncertainty negligible
    K = 8120.276236 * 10 ** -10  # K/(h_bar c)^6 [GeV^-4 s]  Uncertainty negligible
    g_v = 1
    delta_vr = uncertainties.ufloat(0.02467, 0.00022)
    # See equation 14 in PhysRevC.107.015502
    # V_ud = sqrt{K/[G_f^2 g_v^2 (1+delta_vr) Ft_mirror (1 + fa/fv rho^2)]}

    V_uds_rho_only = unumpy.sqrt(K / (Gf ** 2 * g_v ** 2 * (delta_vr.n + 1) * ft_mirrors * (1 + fa_fvs * rhos ** 2)))
    V_uds = unumpy.sqrt(K / (Gf ** 2 * g_v ** 2 * (delta_vr + 1) * ft_mirror_errors * (1 + fa_fv_errors * rhos ** 2)))
    print(list(zip(V_uds_rho_only, ['n', '3H', '11C', '13N', '15O', '17F', '19Ne'])))
    print(list(zip(V_uds, ['n', '3H', '11C', '13N', '15O', '17F', '19Ne'])))
    # print(V_uds_rho_only, '\n\n')
    # print(V_uds)

# Can now write this in a report format:
# Start with theory from JTW, transform into recoil energy spectrum, show different shapes for different values of lil_a
# Show Monte Carlo sampling, how I fit to lil_a
# plots for sensitivity to lil_a (as fn of value of lil_a), bring up table with da/a vs dr/r,
# Current values for Ftmirror, delta, etc, and finish with sensitivity to V_ud w&w/o all uncertainties
# ~7-10 pages with plots.  Can then work towards systematics instead of statistics
