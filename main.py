import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import time
import cProfile

# Constants
deg = np.pi / 180  # converts degrees to radiant
mu = 398600.44  # [km3/s2] gravitational parameter
J2 = 0.00108263  # [-] second zonal harmonic
Re = 6378.14  # [km] earth's radius
we = (2 * np.pi + 2 * np.pi / 365.26) / (24 * 3600)  # [rad/s] earth's angular velocity
exp = np.exp(1)


def kep2car(a, e, incl, W_o, wpo, TAo, n_periods, t_length, year, month, day):
    # Propagates the orbit over the specified time interval, transforming
    # the position and velocity vectors into the earth-fixed frame

    v_ones_i = np.ones(incl.shape)

    h = np.sqrt(mu * a * (1 - (e ** 2)))  # [km2/s] angular momentum of orbit
    Eo = 2 * np.arctan(np.tan(TAo / 2) * np.sqrt((1 - e) / (1 + e)))  # [rad] initial eccentric anomaly
    Mo = Eo - e * np.sin(Eo)  # [rad] initial mean anomaly

    v_ones_W = np.ones(W_o.shape)

    p = a * (1 - e ** 2)

    wpdot = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    Wdot = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)
    Mdott = np.sqrt(mu / a ** 3) * (
                1 - 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(1 - e ** 2) * (3 / 2 * np.sin(incl) ** 2 - 1))

    om_d = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    OM_d = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)
    # M_d = np.sqrt(mu/a**3)* (1 - 3/2 * J2 * (Re/p)**2*np.sqrt(1-e**2) * (3/2*np.sin(incl)**2 - 1))

    W_dot = Wdot * v_ones_W

    T = 2 * np.pi / (wpdot + Mdott)  # [s] period of the orbit
    T_r = n_periods * T

    # print('T_r', T_r)

    to = Mo * (T / (2 * np.pi))  # [s] initial time for the ground track
    tf = to + T_r  # [s] final time for the ground track

    n_step = np.ceil(T_r / t_length).astype(int)  # [steps]
    n_step_max = np.max(n_step)
    step_l = T_r / n_step_max

    times = np.array([np.linspace(to, tf, n_step_max)])  # [s] times at which ground track is plotted
    M = Mo + Mdott * times
    W = W_o + W_dot * times  # RAAN

    wp = wpo + wpdot * times
    E = kepler_E(e, M)
    TA = 2 * np.arctan(np.tan(E / 2) * np.sqrt((1 + e) / (1 - e)))

    # nu = TAo + M_d * times;
    nu = TA
    om = wpo + om_d * times

    #g0 = g0_fun(2022, 4, 1)
    g0 = g0_fun(year, month, day)

    th = we * (times - to) + g0
    # th = 0
    OM = W_o + OM_d * times

    const = (a * (1 - e ** 2)) / (1 + e * np.cos(nu))

    r1 = const * (np.cos(incl) * np.sin(nu + om) * np.sin(th - OM) + np.cos(nu + om) * np.cos(th - OM))
    r2 = const * (np.cos(incl) * np.sin(nu + om) * np.cos(th - OM) - np.cos(nu + om) * np.sin(th - OM))
    r3 = const * (np.sin(incl) * np.sin(nu + om) * v_ones_W)

    r = np.moveaxis(np.concatenate((r1, r2, r3), axis=0), 1, -1)

    v1 = const * ((Mdott + wpdot) * (
                np.cos(incl) * np.cos(nu + om) * np.sin(th - OM) - np.sin(nu + om) * np.cos(th - OM)) + (we - OM_d) * (
                              np.cos(incl) * np.sin(nu + om) * np.cos(th - OM) - np.cos(nu + om) * np.sin(th - OM)))
    v2 = const * ((Mdott + wpdot) * (
                np.cos(incl) * np.cos(nu + om) * np.cos(th - OM) + np.sin(nu + om) * np.sin(th - OM)) - (we - OM_d) * (
                              np.cos(incl) * np.sin(nu + om) * np.sin(th - OM) + np.cos(nu + om) * np.cos(th - OM)))
    v3 = const * ((Mdott + wpdot) * (np.sin(incl) * np.cos(nu + om) * v_ones_W))

    v = np.moveaxis(np.concatenate((v1, v2, v3), axis=0), 1, -1)

    v1 = np.moveaxis(v1, 1, -1)
    v2 = np.moveaxis(v2, 1, -1)
    v3 = np.moveaxis(v3, 1, -1)

    return r, v, step_l, W, wp, TA, th, M, times


def kepler_E(e, M):
    # Using Newton's method, solve Kepler's equation: E - e*sin(E) = M
    # for the eccentric anomaly E, given eccentricity and mean anomaly

    # M [rad] mean anomaly
    # e [-] eccentricity
    # E [rad] eccentric anomaly

    error = 1e-8  # Error tolerance
    E = np.ones(M.shape)
    ratio = np.ones(M.shape)

    # Select starting value for E

    E[:] = M - e / 2
    E[M < np.pi] = M[M < np.pi] + e / 2

    # if M < np.pi:
    #    E = M + e / 2
    # else:
    #    E = M - e / 2

    while np.abs(np.max(ratio)) > error:
        ratio = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - ratio

    return E


def tau2a(tau, e, inc, J2, R_e, om_d_e, mu):
    err = 10000 * np.ones(np.array([inc]).shape)
    n_max = 1000
    n = 0
    a = 10000 * np.ones(np.array([inc]).shape)
    while (np.abs(np.max(err)) > 0.001) & (n < n_max):
        p = a * (1 - e ** 2)
        om_d = 3 / 2 * J2 * (R_e / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(inc) ** 2)
        OM_d = -3 / 2 * J2 * (R_e / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(inc)
        M_d = np.sqrt(mu / a ** 3) * (
                1 - 3 / 2 * J2 * (R_e / p) ** 2 * np.sqrt(1 - e ** 2) * (3 / 2 * np.sin(inc) ** 2 - 1))

        fun = (om_d + M_d) / (om_d_e - OM_d) - tau
        dfun = ((3 * mu * ((3 * J2 * R_e ** 2 * ((3 * np.sin(inc) ** 2) / 2 - 1)) / (
                    2 * a ** 2 * (1 - e ** 2) ** (3 / 2)) - 1)) / (2 * a ** 4 * (mu / a ** 3) ** (1 / 2)) + (
                            3 * J2 * R_e ** 2 * (mu / a ** 3) ** (1 / 2) * ((5 * np.sin(inc) ** 2) / 2 - 2)) / (
                            a ** 3 * (e ** 2 - 1) ** 2) + (
                            3 * J2 * R_e ** 2 * (mu / a ** 3) ** (1 / 2) * ((3 * np.sin(inc) ** 2) / 2 - 1)) / (
                            a ** 3 * (1 - e ** 2) ** (3 / 2)) + (
                            9 * J2 * R_e ** 2 * mu * ((5 * np.sin(inc) ** 2) / 2 - 2)) / (
                            4 * a ** 6 * (e ** 2 - 1) ** 2 * (mu / a ** 3) ** (1 / 2))) / (
                           om_d_e + (3 * J2 * R_e ** 2 * np.cos(inc) * (mu / a ** 3) ** (1 / 2)) / (
                               2 * a ** 2 * (e ** 2 - 1) ** 2)) - (((3 * J2 * R_e ** 2 * np.cos(inc) * (
                    mu / a ** 3) ** (1 / 2)) / (a ** 3 * (e ** 2 - 1) ** 2) + (9 * J2 * R_e ** 2 * mu * np.cos(inc)) / (
                                                                                4 * a ** 6 * (e ** 2 - 1) ** 2 * (
                                                                                    mu / a ** 3) ** (1 / 2))) * (((
                                                                                                                              3 * J2 * R_e ** 2 * (
                                                                                                                                  (
                                                                                                                                              3 * np.sin(
                                                                                                                                          inc) ** 2) / 2 - 1)) / (
                                                                                                                              2 * a ** 2 * (
                                                                                                                                  1 - e ** 2) ** (
                                                                                                                                          3 / 2)) - 1) * (
                                                                                                                             mu / a ** 3) ** (
                                                                                                                             1 / 2) + (
                                                                                                                             3 * J2 * R_e ** 2 * (
                                                                                                                                 mu / a ** 3) ** (
                                                                                                                                         1 / 2) * (
                                                                                                                                         (
                                                                                                                                                     5 * np.sin(
                                                                                                                                                 inc) ** 2) / 2 - 2)) / (
                                                                                                                             2 * a ** 2 * (
                                                                                                                                 e ** 2 - 1) ** 2))) / (
                           om_d_e + (3 * J2 * R_e ** 2 * np.cos(inc) * (mu / a ** 3) ** (1 / 2)) / (
                               2 * a ** 2 * (e ** 2 - 1) ** 2)) ** 2

        a_new = a - fun / dfun
        err = abs(a_new - a)

        n = n + 1

        a = a_new

    print('Semimajor axis calculated \n')

    return a


def ra_and_dec_from_r(r):
    # calculates the right ascension and the declination (latitude)
    # from the geocentric equatorial position vector

    l = r[0, :] / LA.norm(r, axis=0)  # direction cosine
    m = r[1, :] / LA.norm(r, axis=0)  # direction cosine
    n = r[2, :] / LA.norm(r, axis=0)  # direction cosine

    dec = np.arcsin(n)  # [rad] declination (latitude)

    ra = 2 * np.pi - np.arccos(l / np.cos(dec))  # [rad] right ascension (longitude)

    ra[m > 0] = np.arccos(l[m > 0] / np.cos(dec[m > 0]))  # [rad] right ascension (longitude)

    dec = np.degrees(dec)
    ra = np.degrees(ra)

    return ra, dec


def g_t(a, r_rel, v_rel, f_acr, f_alo):
    v_ones = np.ones((1, 1, 1, 1))

    eta = a / Re

    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))

    a_alfa = a_alfa.T * v_ones
    a_beta = a_beta.T * v_ones

    u_r = r_rel / LA.norm(r_rel, axis=0)
    u_v = v_rel / LA.norm(v_rel, axis=0)

    hh = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = hh / LA.norm(hh, axis=0)
    yy = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = yy / LA.norm(yy, axis=0)

    r_rel_M = np.cos(a_alfa) * u_r - np.sin(a_alfa) * u_h
    r_rel_M = r_rel_M / LA.norm(r_rel_M, axis=0)
    (ra_M, dec_M) = ra_and_dec_from_r(r_rel_M)

    r_rel_m = np.cos(-a_alfa) * u_r - np.sin(-a_alfa) * u_h
    r_rel_m = r_rel_m / LA.norm(r_rel_m, axis=0)
    (ra_m, dec_m) = ra_and_dec_from_r(r_rel_m)

    r_rel_N = np.cos(a_beta) * u_r + np.sin(a_beta) * u_y
    r_rel_N = r_rel_N / LA.norm(r_rel_N, axis=0)
    (ra_N, dec_N) = ra_and_dec_from_r(r_rel_N)

    r_rel_n = np.cos(-a_beta) * u_r + np.sin(-a_beta) * u_y
    r_rel_n = r_rel_n / LA.norm(r_rel_n, axis=0)
    (ra_n, dec_n) = ra_and_dec_from_r(r_rel_n)

    print(ra_n.shape)

    return ra_M, ra_m, ra_N, ra_n, dec_M, dec_m, dec_N, dec_n


def latlon2car(lat, lon, R):
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    r = np.array([x, y, z])

    print('Target position vector calculated \n')
    return r


def dot_p(r_sat, r_t):
    if np.ndim(r_sat) == 4:
        ang = np.einsum('mois,mt->tois', r_sat, r_t)
    elif np.ndim(r_sat) == 3:
        ang = np.einsum('mos,mt->tos', r_sat, r_t)
    elif np.ndim(r_sat) == 2:
        ang = np.einsum('ms,mt->ts', r_sat, r_t)

    # ang = np.einsum('mois,mt->tois', r_sat, r_t)

    return ang


def unit_v(v):
    u_v = v / LA.norm(v, axis=0)  # direction cosine

    return u_v


def filt_an(a, r, v, r_t, f_acr, f_alo):
    v_ones = np.ones((1, 1, 1, 1))
    eta = a / Re
    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T * v_ones

    u_r = unit_v(r)
    u_r_t = unit_v(r_t)

    ang = np.arccos(dot_p(u_r, u_r_t))
    filt_steps = np.absolute(ang) <= a_alfa

    # print('filt_steps shape: ', filt_steps.shape)
    # print('filt_steps sum: ', np.sum(filt_steps))

    # return matrix, each line vector contains index of non zero cell
    # 0: targets, 1: Om, 2: incl, 3: time_steps
    cov_steps = np.array(np.nonzero(filt_steps[:]))

    # print('cov_steps shape: ', cov_steps.shape)

    n_targets = r_t.shape[1]  # number of targets

    # return unique 3-tuples (target, OM, incl) with recurrences == # timesteps covered by each unique 3-tuple
    # matrix 3 x #timesteps covered in total
    pairs_3, counts_3 = np.unique(cov_steps[0:-1, :], axis=1, return_counts=True)

    # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
    # matrix 2 x #combinations OM/incl satisfied timesteps
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # mask for the 2-tuples present in all targets
    filter_all_targets = counts_2 == n_targets  # add verification coversge requirement

    # return unique 2-tuples (OM, incl) with recurrences == # timesteps covered by each unique 2-tuples
    pairs, counts = np.unique(cov_steps[1:-1, :], axis=1, return_counts=True)

    # max timesteps covered by the tuples present in each target
    max_ts = np.max(counts[filter_all_targets])

    # number of orbits satisfying max cond
    n_max_ts = np.sum(max_ts == counts[filter_all_targets])

    # position max_ts
    pos_max_ts = counts[filter_all_targets] == max_ts

    # pair satisfying all targets condition
    pairs_ok = pairs[:, filter_all_targets]

    # max pair
    max_pair = pairs_ok[:, pos_max_ts]

    print('Maximum number of timestep covered (among all targets): ', max_ts)
    print('Number of orbits (OM, incl) that satisfy max ts: ', n_max_ts)
    print('Best pair (OM, incl): ', max_pair)

    print('Best pair(s) calculated \n')
    return max_pair


def best_pair(a, Re, W_o, incl, max_pairs):
    print('Choosing first best pair')
    pair_index = 0  # first best pair

    ind_W = max_pairs[0, pair_index]
    ind_i = max_pairs[1, pair_index]

    b_W = W_o[ind_W, 0]
    b_i = incl[ind_i]
    b_a = a[0, ind_i]

    print('Chosen OM: ', np.around(np.rad2deg(b_W), 2), '[deg]')
    print('Chosen incl: ', np.around(np.rad2deg(b_i), 2), '[deg]')
    print('Chosen a: ', np.around(b_a, 2), '[km] (h: ', np.around(b_a - Re, 2), '[km])')
    # print('\n')
    return ind_W, ind_i, b_a


def plot_exp_track(r, lat_t, lon_t, N_p):
    deg = np.pi / 180  # converts degrees to radiant
    ra, dec = ra_and_dec_from_r(r)

    ra_u = np.unwrap(ra * deg) / deg

    ind = np.arange(N_p)

    fig, ax = plt.subplots(figsize=(20, 9))
    ax.plot(np.unwrap(ra * deg) / deg, dec)
    ax.vlines(x=ind * 360, ymin=-90, ymax=90, ls=':')
    for i in ind[:-1]:
        ax.scatter((lon_t + i * 2 * np.pi) / deg, lat_t / deg, s=20, marker='.')

    return


def pair_pop(N_s, N_p, N_d, Om_0, M_0, incl, a, e):
    p = a * (1 - e ** 2)
    wpdot = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    Wdot = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)
    Mdott = np.sqrt(mu / a ** 3) * (
            1 - 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(1 - e ** 2) * (3 / 2 * np.sin(incl) ** 2 - 1))
    # theta = we * (times - to)
    we = (2 * np.pi + 2 * np.pi / 365.26) / (24 * 3600)  # [rad/s] earth's angular velocity
    print('incl', incl)
    print('wpdot', wpdot.shape)
    print('Wdot', Wdot.shape)
    print('Mdott', Mdott.shape)

    i = np.arange(N_s)

    # Om_k = 2*np.pi * i * N_d / N_s + Om_0
    # M_k = -M_0 + N_p/N_d * (Om_k - Om_0)

    print(Om_0.shape)
    print(we + Wdot)
    Om_k = Om_0 - (we + Wdot).T * i

    M_k = M_0 + (Mdott + wpdot).T * i

    print(Om_k.shape)
    print(M_k.shape)

    check = N_p * (Om_k - Om_0) - N_d * (M_k - M_0)

    if np.sum(check) == 0:
        print('Pairs ok!')
    else:
        print('!!!!! CHECKED FAILED !!!!!')
        print('Sum(check): ', np.sum(check), '\n')

    Om_k_d = np.rad2deg(Om_k)
    M_k_d = np.rad2deg(M_k)

    # np.set_printoptions(precision=3)
    # np.set_printoptions(suppress=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Om_k_d % 360, M_k_d % 360, s=10, marker='.')
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 360), ylim=(0, 360));

    # print('(Om-M)-pairs calculated:')
    # print(Om_k_d % 360)
    # print(M_k_d % 360, '\n')

    return Om_k, M_k


def sat_pop(a, e, incl, W_o_v, wpo, TAo_v, n_periods, t_length, n_steps):
    n_pairs = TAo_v.shape[0]

    r_v = np.empty((3, n_pairs, n_steps))
    v_v = np.empty((3, n_pairs, n_steps))
    # np.array([incl])
    # np.array([W_o_v[i]])

    for i in np.arange(n_pairs):
        (r, v, step_l, W, wp, TA, theta, M, times) = kep2car(a, e, incl, W_o_v[i], wpo, TAo_v[i], n_periods, t_length)
        r_v[:, i, :] = r
        v_v[:, i, :] = v

    print('Satellites position vectors calculated')

    return r_v, v_v


def projections(r, v, r_t):
    u_r = unit_v(r)
    u_v = unit_v(v)
    u_r_t = unit_v(r_t)

    print('new unit vectors calculated')

    u_h = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = u_h / LA.norm(u_h, axis=0)
    u_y = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = u_y / LA.norm(u_y, axis=0)

    print('new system reference calculated')

    # target projection on new system of reference

    p1 = dot_p(u_r, u_r_t)
    p2 = dot_p(u_y, u_r_t)
    p3 = dot_p(u_h, u_r_t)

    print('projections calculated')

    return p1, p2, p3


def filt_anV2(a, r, v, r_t, f_acr, f_alo):
    v_ones = np.ones((1, 1, 1, 1))
    eta = a / Re

    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T * v_ones

    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
    a_beta = a_beta.T * v_ones

    filt_steps = filt_steps_fun(r, v, r_t, a_alfa, a_beta)

    # ang = np.arccos(dot_p(u_r, u_r_t))
    # filt_steps = np.absolute(ang) <= a_alfa

    # print('filt_steps shape: ', filt_steps.shape)
    # print('filt_steps sum: ', np.sum(filt_steps))

    # return matrix, each line vector contains index of non zero cell
    # 0: targets, 1: Om, 2: incl, 3: time_steps
    cov_steps = np.array(np.nonzero(filt_steps[:]))
    # print(cov_steps)

    # print('cov_steps shape: ', cov_steps.shape)

    n_targets = r_t.shape[1]  # number of targets

    # return unique 3-tuples (target, OM, incl) with recurrences == # timesteps covered by each unique 3-tuple
    # matrix 3 x #timesteps covered in total
    pairs_3, counts_3 = np.unique(cov_steps[0:-1, :], axis=1, return_counts=True)

    # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
    # matrix 2 x #combinations OM/incl satisfied timesteps
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # mask for the 2-tuples present in all targets
    filter_all_targets = counts_2 == n_targets  # add verification coversge requirement

    # return unique 2-tuples (OM, incl) with recurrences == # timesteps covered by each unique 2-tuples
    pairs, counts = np.unique(cov_steps[1:-1, :], axis=1, return_counts=True)

    # max timesteps covered by the tuples present in each target
    # print(counts[filter_all_targets])
    max_ts = np.max(counts[filter_all_targets])

    # number of orbits satisfying max cond
    n_max_ts = np.sum(max_ts == counts[filter_all_targets])

    # position max_ts
    pos_max_ts = counts[filter_all_targets] == max_ts

    # pair satisfying all targets condition
    pairs_ok = pairs[:, filter_all_targets]

    # max pair
    max_pair = pairs_ok[:, pos_max_ts]

    print('Maximum number of timestep covered (among all targets): ', max_ts)
    print('Number of orbits (OM, incl) that satisfy max ts: ', n_max_ts)
    print('Best pair (OM, incl): ', max_pair)

    print('Best pair(s) calculated \n')

    return max_pair


def read_targets():
    lat_t, lon_t = np.loadtxt("constellation_targets.csv", delimiter=',', usecols=(1, 2), unpack=True)

    # lon_t = lon_t + 180

    lat_t = np.radians(lat_t)
    lon_t = np.radians(lon_t)

    return lon_t, lat_t


def filt_steps_fun(r, v, r_t, a_alfa, a_beta):
    dist_tol = 20  # [km] error tolerance in the cone sensor
    alf_tol = np.arctan(dist_tol / Re)

    p1, p2, p3 = projections(r, v, r_t)

    mask_p1 = p1 > 0

    # filt_steps_al = np.full(p1.shape, False)
    # filt_steps_ac = np.full(p1.shape, False)
    # filt_steps = np.full(p1.shape, False)

    # along track
    # psi = np.arctan2(p2, p1)
    # filt_steps_al = np.absolute(psi) <= a_beta

    filt_steps_al = np.absolute(p2) / p1 <= np.tan(a_beta)
    filt_steps_al[~mask_p1] = False

    print('along filter ok', np.sum(filt_steps_al))

    # across track
    # phi = np.arctan2(p3, p1)
    # filt_steps_ac = np.absolute(phi) <= a_alfa

    filt_steps_ac = np.absolute(p3) / p1 <= np.tan(a_alfa - alf_tol)
    filt_steps_ac[~mask_p1] = False

    print('across filter ok ', np.sum(filt_steps_ac))

    filt_steps = np.logical_and(filt_steps_al, filt_steps_ac)

    print('total filter ok', np.sum(filt_steps))

    return filt_steps


def filt_anV3(a, r, v, r_t, f_acr, f_alo, rev_time, n_step):
    v_ones = np.ones((1, 1, 1, 1))
    eta = a / Re

    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T * v_ones

    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
    a_beta = a_beta.T * v_ones

    filt_steps = filt_steps_fun(r, v, r_t, a_alfa, a_beta)

    # ang = np.arccos(dot_p(u_r, u_r_t))
    # filt_steps = np.absolute(ang) <= a_alfa

    # print('filt_steps shape: ', filt_steps.shape)
    # print('filt_steps sum: ', np.sum(filt_steps))

    # return matrix, each line vector contains index of non zero cell
    # 0: targets, 1: Om, 2: incl, 3: time_steps
    cov_steps = np.array(np.nonzero(filt_steps[:]))
    # print(cov_steps)

    # print('cov_steps shape: ', cov_steps.shape)

    cov_steps_rt = cov_steps.copy()  # revisit time

    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = 24 / time_wind  # [] length of a time window in terms of timesteps

    cov_steps_rt[-1, :] = np.floor(cov_steps[-1, :] / (n_step / (time_wind_step)))

    pairs_rt, counts_rt = np.unique(cov_steps_rt, axis=1, return_counts=True)

    n_targets = r_t.shape[1]  # number of targets

    # return unique 3-tuples (target, OM, incl) with recurrences == # time windows covered by each unique 3-tuple
    # matrix 3 x # timewindow covered in total
    ###pairs_3, counts_3 = np.unique(cov_steps[0:-1,:], axis=1, return_counts=True)

    pairs_3, counts_3 = np.unique(pairs_rt[0:-1, :], axis=1, return_counts=True)

    # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
    # matrix 2 x #combinations OM/incl satisfied time windows
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # mask for the 2-tuples present in all targets
    filter_all_targets = counts_2 == n_targets

    if filter_all_targets.any():
        print("All targets are covered by in at least one time window. Coverage requirement ok")
    else:
        print(
            "ERROR!!! No tuple can cover all satellite in at least one window. Try reducing ang-steps or increasing angle sensor camera!")

    # return unique 2-tuples (OM, incl) with recurrences == # time windows covered by each unique 2-tuples
    ###pairs, counts = np.unique(cov_steps[1:-1,:], axis=1, return_counts=True)
    pairs, counts = np.unique(pairs_rt[1:-1, :], axis=1, return_counts=True)

    # max timesteps covered by the tuples present in each target
    max_ts = np.max(counts[filter_all_targets])

    # number of orbits satisfying max cond
    n_max_ts = np.sum(max_ts == counts[filter_all_targets])

    # position max_ts
    pos_max_ts = counts[filter_all_targets] == max_ts

    # pair satisfying all targets condition
    pairs_ok = pairs[:, filter_all_targets]

    # max pair
    max_pair = pairs_ok[:, pos_max_ts]

    print('Maximum number of timestep covered (among all targets): ', max_ts)
    print('Number of orbits (OM, incl) that satisfy max ts: ', n_max_ts)
    print('Best pair (OM, incl): ', max_pair)

    print('Best pair(s) calculated \n')

    return max_pair


def j0_fun(year, month, day):
    # calculates the julian day number at 0 UT

    j0 = 367 * year - np.fix(7 * (year + np.fix((month + 9) / 12)) / 4) + np.fix(275 * month / 9) + day + 1721013.5

    return j0


def g0_fun(year, month, day):
    j0 = j0_fun(year, month, day)

    j = (j0 - 2451545) / 36525

    g0 = 100.4606184 + 36000.7704 * j + 0.000387933 * (j ** 2) - 2.583 * (10 ** -8) * (j ** 3)

    g0 = np.radians(g0)

    nn = np.floor(g0 / (2 * np.pi))

    g0 = g0 - nn * 2 * np.pi

    return g0


def cov_steps_fun(a, r, v, r_t, f_acr, f_alo, rev_time, n_step):
    v_ones = np.ones((1, 1, 1, 1))
    eta = a / Re

    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T * v_ones

    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
    a_beta = a_beta.T * v_ones

    filt_steps = filt_steps_fun(r, v, r_t, a_alfa, a_beta)

    # ang = np.arccos(dot_p(u_r, u_r_t))
    # filt_steps = np.absolute(ang) <= a_alfa

    # print('filt_steps shape: ', filt_steps.shape)
    # print('filt_steps sum: ', np.sum(filt_steps))

    # return matrix, each line vector contains index of non zero cell
    # 0: targets, 1: Om, 2: incl, 3: time_steps
    cov_steps = np.array(np.nonzero(filt_steps[:]))
    # print(cov_steps)

    # print('cov_steps shape: ', cov_steps.shape)

    cov_steps_rt = cov_steps.copy()  # revisit time

    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = 24 / time_wind  # [] length of a time window in terms of timesteps

    cov_steps_rt[-1, :] = np.floor(cov_steps[-1, :] / (n_step / (time_wind_step)))

    pairs_rt, counts_rt = np.unique(cov_steps_rt, axis=1, return_counts=True)
    # print(pairs_rt)
    n_targets = r_t.shape[1]  # number of targets

    # return unique 3-tuples (target, OM, incl) with recurrences == # time windows covered by each unique 3-tuple
    # matrix 3 x # timewindow covered in total

    pairs_3, counts_3 = np.unique(pairs_rt[0:-1, :], axis=1, return_counts=True)

    # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
    # matrix 2 x #combinations OM/incl satisfied time windows
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # mask for the 2-tuples present in all targets
    ##filter_all_targets = counts_2 == n_targets   

    ##if filter_all_targets.any():
    ##    print("All targets are covered by in at least one time window. Coverage requirement ok")
    ##else:
    ##    print("ERROR!!! No tuple can cover all satellite in at least one window. Try reducing ang-steps or increasing angle sensor camera!")

    # return unique 2-tuples (OM, incl) with recurrences == # time windows covered by each unique 2-tuples
    ###pairs, counts = np.unique(cov_steps[1:-1,:], axis=1, return_counts=True)
    ##pairs, counts = np.unique(pairs_rt[1:-1,:], axis=1, return_counts=True)

    # max timesteps covered by the tuples present in each target
    ##max_ts = np.max(counts[filter_all_targets])  

    # number of orbits satisfying max cond
    ##n_max_ts = np.sum(max_ts == counts[filter_all_targets])

    # position max_ts
    ##pos_max_ts = counts[filter_all_targets] == max_ts

    # pair satisfying all targets condition
    ##pairs_ok = pairs[:, filter_all_targets]

    # max pair
    ##max_pair = pairs_ok[:, pos_max_ts]

    ##print('Maximum number of timestep covered (among all targets): ', max_ts)
    ##print('Number of orbits (OM, incl) that satisfy max ts: ', n_max_ts)
    ##print('Best pair (OM, incl): ', max_pair)

    ##print('Best pair(s) calculated \n')

    return cov_steps


def set_cover_prblm_targets(cov_steps, r_t, rev_time, n_step):
    cov_steps_rt = cov_steps.copy()  # revisit time

    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = 24 / time_wind  # [] length of a time window in terms of timesteps

    cov_steps_rt[-1, :] = np.floor(cov_steps[-1, :] / (n_step / (time_wind_step)))
    pairs_rt, counts_rt = np.unique(cov_steps_rt, axis=1, return_counts=True)

    n_targets = r_t.shape[1]  # number of targets

    # return unique 3-tuples (target, OM, incl) with recurrences == # time windows covered by each unique 3-tuple
    # matrix 3 x # timewindows covered in total
    pairs_3, counts_3 = np.unique(pairs_rt[0:-1, :], axis=1, return_counts=True)

    # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
    # matrix 2 x #combinations OM/incl satisfied time windows
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # return unique incl with number of targets covered -> up for now one inclination should cover all targets
    # counts_1 should be number of Oms
    pairs_1, counts_1 = np.unique(pairs_3[(0, 2), :], axis=1, return_counts=True)
    pairs_11, counts_11 = np.unique(pairs_1[1, :],
                                    return_counts=True)  # counts_11: number of targets covered by each inclination

    admissible_incl = pairs_11[counts_11 == n_targets]  # admissible inclinations must cover all targets
    mask_adm_incl = np.isin(pairs_rt[2, :], admissible_incl)

    ## filtering again the population with the admissible inclination
    pairs_3, counts_3 = np.unique(pairs_rt[0:-1, mask_adm_incl], axis=1, return_counts=True)
    pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

    # return unique 2-tuples (OM, incl) with recurrences == # time windows covered by each unique 2-tuples
    pairs, counts = np.unique(pairs_rt[1:-1, mask_adm_incl], axis=1, return_counts=True)

    max_cov_targets = np.amax(counts_2)  # obtain max targets covered by all tuples OM-i
    vec_cov_timeslots = counts[
        counts_2 == max_cov_targets]  # vector of covered timeslots by tuples with max cov targets
    max_vec_cov_timeslots = np.amax(vec_cov_timeslots)

    print('max cov targets', max_cov_targets)

    candidate_pairs = pairs[:,
                      counts_2 == max_cov_targets]  # filtering pairs with most covered targets. output matrix 2 (om, incl) x candidate pairs
    possible_pairs = candidate_pairs[:,
                     vec_cov_timeslots == max_vec_cov_timeslots]  # filtering pairs from last with most covered timeslots

    # choosing pairs with most recurring inclination angle
    unique_incl, count_incl = np.unique(possible_pairs[1, :], return_counts=True)

    max_count_incl = np.amax(count_incl)
    best_incl = unique_incl[count_incl == max_count_incl]
    best_incl = best_incl[0]  # choosing first best incl in case there are equally as good
    mask_incl = pairs_rt[2, :] == best_incl

    Om_vec = np.empty(1000)

    possible_Om = possible_pairs[0, possible_pairs[1, :] == best_incl]  ### to review best incl for future constellation

    best_Om = possible_Om[0]  # choosing first best possible Om
    mask_Om = pairs_rt[1, :] == best_Om
    Om_vec[0] = best_Om

    filtered_pairs_rt = pairs_rt[:, np.logical_and(mask_Om, mask_incl)]
    chosen_targets = np.unique(filtered_pairs_rt[0, :])
    # print('chosen_targets', chosen_targets)

    mask_uncovered_targets = np.logical_not(np.isin(pairs_rt[0, :], chosen_targets))
    mask_tot = np.logical_and(mask_incl, mask_uncovered_targets)

    tot_uncov_targets = np.sum(mask_tot)

    pairs_rt_updated = pairs_rt[:, mask_tot]

    cont = 0
    max_cont = 10000
    while np.logical_and(cont < max_cont, tot_uncov_targets != 0):
        pairs_3, counts_3 = np.unique(pairs_rt_updated[0:-1, :], axis=1, return_counts=True)

        # return unique 2-tuples from pairs_3 (OM, incl) with recurrences == # 2-tuple present in each target
        # matrix 2 x #combinations OM/incl satisfied time windows
        pairs_2, counts_2 = np.unique(pairs_3[1:, ], axis=1, return_counts=True)

        # return unique 2-tuples (OM, incl) with recurrences == # time windows covered by each unique 2-tuples
        pairs, counts = np.unique(pairs_rt_updated[1:-1, :], axis=1, return_counts=True)

        max_cov_targets = np.amax(counts_2)  # obtain max targets covered by all tuples OM-i
        print('max cov targets', max_cov_targets)
        vec_cov_timeslots = counts[
            counts_2 == max_cov_targets]  # vector of covered timeslots by tuples with max cov targets
        max_vec_cov_timeslots = np.amax(vec_cov_timeslots)

        candidate_pairs = pairs[:,
                          counts_2 == max_cov_targets]  # filtering pairs with most covered targets. output matrix 2 (om, incl) x candidate pairs
        possible_pairs = candidate_pairs[:,
                         vec_cov_timeslots == max_vec_cov_timeslots]  # matrix 2 x possible pairs. Row of incl already chosen one before

        best_Om = possible_pairs[
            0, 0]  # choosing first one possible Om. incl already chosen, max timeslots already chosen
        Om_vec[cont + 1] = best_Om  # saving picked Om in the vector
        mask_Om = np.logical_or(mask_Om, pairs_rt[1, :] == best_Om)

        filtered_pairs_rt = pairs_rt[:, np.logical_and(mask_Om, mask_incl)]
        chosen_targets = np.unique(filtered_pairs_rt[0, :])
        # print('chosen_targets', chosen_targets)

        mask_uncovered_targets = np.logical_not(
            np.isin(pairs_rt[0, :], chosen_targets))  # if false, then the column (target) still need to be covered
        mask_tot = np.logical_and(mask_incl, mask_uncovered_targets)
        tot_uncov_targets = np.sum(mask_tot)

        pairs_rt_updated = pairs_rt[:, mask_tot]

        cont = cont + 1

    Om_index_vec = Om_vec[0:cont + 1].astype(int)
    incl_index_vec = best_incl * np.ones(Om_index_vec.shape).astype(int)

    ## verify coverage
    mask_Om_index = np.isin(pairs_rt[1, :], Om_index_vec)
    mask_in_index = np.isin(pairs_rt[2, :], incl_index_vec)

    targets_cov = np.unique(pairs_rt[0, np.logical_and(mask_Om_index, mask_in_index)])

    check = targets_cov == np.arange(n_targets)

    if np.all(check):
        print('Ok! All target locations are covered')
    else:
        print('!!!!!! ERROR !!!!!! Not all targets are covered')
        print('Targets covered:', targets_cov)

    return Om_index_vec, incl_index_vec


def pair_pop_mod(N_s, N_p, N_d, Om_0, M_0, incl, a, e, n_step, step_l, tim, n_periods):
    p = a * (1 - e ** 2)
    wpdot = 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * (2 - 5 / 2 * np.sin(incl) ** 2)
    Wdot = -3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(mu / a ** 3) * np.cos(incl)
    Mdott = np.sqrt(mu / a ** 3) * (
            1 - 3 / 2 * J2 * (Re / p) ** 2 * np.sqrt(1 - e ** 2) * (3 / 2 * np.sin(incl) ** 2 - 1))
    # theta = we * (times - to)
    we = (2 * np.pi + 2 * np.pi / 365.26) / (24 * 3600)  # [rad/s] earth's angular velocity

    T = 2 * np.pi / (wpdot + Mdott)  # [s] period of the orbit
    T_r = n_periods * T

    # to = Mo * (T / (2 * np.pi))  # [s] initial time for the ground track
    # th_0 = we * to
    # theta = we * (times - to)

    # i = np.arange(n_step)
    # i = np.arange(n_step)*step_l
    ii = tim
    # Om_k = 2*np.pi * i * N_d / N_s + Om_0
    # M_k = -M_0 + N_p/N_d * (Om_k - Om_0)

    wpp = wpdot * ii
    taa = Mdott * ii
    thh = we * ii

    Om_k = Om_0 - (we - Wdot) * ii
    M_k = M_0 + (Mdott + wpdot) * ii

    # Om_k = Om_0 - i * 2*np.pi * N_d/N_s
    # M_k = M_0 + i * 2*np.pi * N_p/N_s

    check = N_p * (Om_k - Om_0) + N_d * (M_k - M_0)
    # print(check)

    if np.sum(check) < 10 ** -8:
        print('Pairs ok!')
    else:
        print('!!!!! CHECKED FAILED !!!!!')
        print('Sum(check): ', np.sum(check), '\n')

    Om_k_d = np.rad2deg(Om_k)
    M_k_d = np.rad2deg(M_k)

    # np.set_printoptions(precision=3)
    # np.set_printoptions(suppress=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Om_k_d % 360, M_k_d % 360, s=10, marker='.')
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 360), ylim=(0, 360));

    # print('(Om-M)-pairs calculated:')
    # print(Om_k_d % 360)
    # print(M_k_d % 360, '\n')
    return Om_k, M_k, wpp, taa, thh, ii


def sat_pop_mod(a, e, incl, W_o_v, wpo, TAo_v, n_periods, t_length, n_steps, year, month, day):
    n_pairs = W_o_v.shape[0]

    r_v = np.empty((3, n_pairs, n_steps))
    v_v = np.empty((3, n_pairs, n_steps))
    # np.array([incl])
    # np.array([W_o_v[i]])

    for i in np.arange(n_pairs):
        (r, v, step_l, W, wp, TA, theta, M, times) = kep2car(a, e, incl, W_o_v[i], TAo_v[i], wpo, n_periods, t_length, year, month, day)
        r_v[:, i, :] = r
        v_v[:, i, :] = v

    print('Satellites position vectors calculated')

    return r_v, v_v


def const_rv_vec(Om_index_vec, incl_index_vec, N_s, N_p, N_d, W_o, incl, a, e, n_step, step_l, times, wpo, n_periods, year, month, day):
    n_subconst = Om_index_vec.shape[0]

    r_const = np.empty((3, n_subconst * n_step, n_step))  # 3 coordinates x n. pairs x n. timesteps
    v_const = np.empty((3, n_subconst * n_step, n_step))  # 3 coordinates x n. pairs x n. timesteps
    Om_const = np.empty((1, n_subconst * n_step))
    M_const = np.empty((1, n_subconst * n_step))

    for i in np.arange(n_subconst):
        W_opt = Om_index_vec[i]
        incl_opt = incl_index_vec[i]
        a_b = a[0, incl_opt]

        Om_pop, M_pop, wpp, taa, thh, ii = pair_pop_mod(N_s, N_p, N_d, W_o[W_opt, 0], 0, incl[incl_opt], a_b, e, n_step,
                                                        step_l[0, incl_opt], times[0, :, 0, incl_opt], n_periods)
        # r_v, v_v = sat_pop(a_b, e, incl[ind_i], Om_pop, wpo, M_pop, n_periods, step_l[:, ind_i], n_step)
        r_v, v_v = sat_pop_mod(a_b, e, incl[incl_opt], Om_pop, wpo, M_pop, n_periods, step_l[:, incl_opt], n_step, year, month, day)

        r_const[:, i * n_step: (i + 1) * n_step, :] = r_v
        v_const[:, i * n_step: (i + 1) * n_step, :] = v_v
        Om_const[0, i * n_step: (i + 1) * n_step] = Om_pop
        M_const[0, i * n_step: (i + 1) * n_step] = M_pop

        i = i + 1

    return a_b, r_const, v_const, Om_const, M_const


def filt_pop(a, r, v, r_t, f_acr, f_alo):
    eta = a / Re
    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T

    a_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
    a_beta = a_beta.T

    # p1, p2, p3 = projections(r, v, r_t)

    # u_r = unit_v(r)
    # u_r_t = unit_v(r_t)

    # ang = np.arccos(dot_p(u_r, u_r_t))
    # filt_steps = np.absolute(ang) <= a_alfa

    # along track
    # psi = np.arctan2(p2, p1)

    # filt_steps_al = np.absolute(psi) <= a_beta

    # across track
    # phi = np.arctan2(p3, p1)

    # filt_steps_ac = np.absolute(phi) <= a_alfa

    # filt_steps = np.logical_and(filt_steps_al, filt_steps_ac)

    filt_steps = filt_steps_fun(r, v, r_t, a_alfa, a_beta)

    cov_stepss = np.array(np.nonzero(filt_steps[:]))

    return cov_stepss


def cont_time_check(cov_pop, n_step):
    tar_time_mat = np.delete(cov_pop, 1, axis=0)  # deleting first row (population index)
    pairs, counts = np.unique(tar_time_mat, axis=1, return_counts=True)  # pairs == target-timesteps

    tar_mat = pairs[0, :]  # keeping targets with the number of recurences (timesteps)
    tar, count_time = np.unique(tar_mat, axis=0, return_counts=True)

    if np.all(count_time == n_step):
        print('Continuous coverage possible!')
    else:
        print('Continuous coverage NOT possible!')

        print(tar)
        print(count_time)

    return 0


def cov_uni(rev_time, step_l, N_d, n_step, n_targets):
    # ASSUMING same revisit time for all targets -> same time windows for all -> same length & number

    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = N_d * 24 / time_wind  # [] length of a time window in terms of timesteps

    n_time_wds = np.floor(24 * N_d / time_wind)  # number of time windows
    n_time_wds = n_time_wds.astype(int)

    time_wds = np.arange(n_time_wds)  # time windows

    uni_mat = np.zeros((n_targets, n_time_wds))
    uni_mat[:, 0:n_time_wds] = time_wds

    # print(uni_mat)

    print('ok')

    return uni_mat


def cov_set(cov_pop, uni_mat, rev_time, n_step, N_d):
    size_pop = cov_pop.shape[1]
    n_targets = uni_mat.shape[0]

    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = N_d * 24 / time_wind  # [] length of a time window in terms of timesteps

    times_cov = np.floor(cov_pop[2] / (n_step / (time_wind_step)))

    # times_cov = cov_pop[2]
    mask = np.full((size_pop), False, dtype=bool)

    for i in np.arange(n_targets):
        times_req = uni_mat[i, :]

        mask_target = cov_pop[0, :] == i
        mask_times = np.isin(times_cov, times_req)

        mask_gen = np.logical_and(mask_target, mask_times)

        mask = np.logical_or(mask, mask_gen)

    set_mat = cov_pop[:, mask]
    set_mat[2, :] = np.floor(set_mat[2, :] / (n_step / (time_wind_step)))
    set_mat = np.unique(set_mat, axis=1)

    return set_mat


def cov_probl(set_mat, n_targets):
    n_index, priority = np.unique(set_mat[1, :], axis=0, return_counts=True)
    # print('priority shape: ', priority.shape)
    # print('sat number shape: ', n_index.shape)
    # print('sat number: ', n_index)
    # print('sat priority: ', priority, '\n')

    n_sets = n_index.shape[0]
    # n_max = n_sets  # number max iterations
    n_max = n_sets
    const_vec = np.empty(n_sets)  # constellation vector == satellite number picked
    pry_vec = np.empty(n_sets)
    cont = 0

    set_mat_remaining = set_mat
    time_s_rem = set_mat_remaining.shape[1]

    while cont < n_max and time_s_rem > 0:
        n_index, priority = np.unique(set_mat_remaining[1, :], axis=0, return_counts=True)

        pry_max = np.max(priority, where=True)  # max priority
        # print('max priority: ', pry_max)
        pry_vec[cont] = pry_max

        pry_max_i = np.array(np.where(priority == pry_max))
        # pry_max_i = pry_max_i[0,0]                               # max priority [first] index
        pry_max_i = pry_max_i[0, -1]  # max priority [last] index
        # print('max priority [first] index: ', pry_max_i , '\n')

        n_index_best = n_index[pry_max_i]  # best sat #
        # print('best sat #: ', n_index_best)

        # print('index_best: ', n_index_best)

        mask_pi = set_mat_remaining[1, :] == n_index_best

        picked_mat = set_mat_remaining[:, mask_pi]

        # print('picked mat: ',picked_mat)

        # print(np.sum(mask_pi))
        # print(set_mat_remaining[:, mask_pi].shape)
        # print(set_mat_remaining[:, mask_pi])

        mask = np.full((time_s_rem), False, dtype=bool)
        mask_i = 0

        for i in np.arange(n_targets):
            mask_target = set_mat_remaining[0, :] == i
            mask_picked = picked_mat[0, :] == i

            # print('mask_picked: ', mask_picked)

            time_s_rem = set_mat_remaining[2, mask_target]

            # print('set_mat_remaining[2,:].shape : ', set_mat_remaining[2, :].shape)
            # print('set_mat_remaining[2,:] : ', set_mat_remaining[2, :])
            # print('time_s_remaining.shape: ', time_s_rem.shape)
            # print('time_s_rem: ', time_s_rem)

            mask_f = mask_i + time_s_rem.shape[0]

            time_picked = picked_mat[2, mask_picked]

            mask_p = np.isin(time_s_rem, time_picked)  # partial mask

            # print('mask_i: ', mask_i)
            # print('mask_f: ', mask_f)
            mask[mask_i:mask_f] = mask_p
            mask_i = mask_f

        set_mat_remaining = set_mat_remaining[:, ~mask]
        const_vec[cont] = n_index_best

        cont = cont + 1
        time_s_rem = set_mat_remaining.shape[1]

    print('Converged!')
    print('Number of satellites: ', cont)

    const_vec = const_vec[0:cont]
    const_vec = const_vec.astype(int)
    pry_vec = pry_vec[0:cont]
    pry_vec = pry_vec.astype(int)

    return const_vec, pry_vec


def const_info(const_vec, Om_pop, M_pop, n_step):
    # Om_const = Om_pop[0,const_vec,const_vec % n_step]
    # M_const = M_pop[0,const_vec,const_vec % n_step]

    Om_const = Om_pop[0, const_vec]
    M_const = M_pop[0, const_vec]

    Om_k_d_c = np.rad2deg(Om_const)
    M_k_d_c = np.rad2deg(M_const)

    Om_k_d = np.rad2deg(Om_pop)
    M_k_d = np.rad2deg(M_pop)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Om_k_d % 360, M_k_d % 360, s=1, marker='.')
    ax.scatter(Om_k_d_c % 360, M_k_d_c % 360, s=12, marker='.')
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 360), ylim=(0, 360), xlabel='Om [deg]', ylabel='M [deg]');

    return Om_const, M_const


def const_mat_fun(a, e, i, Om, om, th):
    n_sats = Om.shape[0]
    ones_v = np.ones(n_sats).T

    a_v = a * ones_v
    e_v = e * ones_v

    i = np.degrees(i)
    i_v = i * ones_v

    Om = np.degrees(Om).T
    Om = Om % 360

    # om = np.degrees(om)
    # om_v = om * ones_v

    # th = th % 2*np.pi
    # th = np.degrees(th).T

    om = np.degrees(om).T
    om = om % 360

    th = np.degrees(th)
    th_v = th * ones_v

    const_mat = np.empty((n_sats, 6))
    const_mat[:, 0] = a_v
    const_mat[:, 1] = e_v
    const_mat[:, 2] = i_v
    const_mat[:, 3] = Om
    const_mat[:, 4] = om
    const_mat[:, 5] = th_v

    return const_mat


def const_verfy(cov_pop, const_vec, rev_time, n_step, n_targets, N_d):
    rev_time_h = rev_time / 60  # [h] hours fraction of revisit time
    time_wind = rev_time_h / 2  # [h] time windows -> ensuring worse case scenario
    time_wind_step = 24 * N_d / time_wind  # [] length of a time window in terms of timesteps

    mask = np.isin(cov_pop[1, :], const_vec)

    const_pop = cov_pop[:, mask]
    const_pop[2, :] = np.floor(const_pop[2, :] / (n_step / (time_wind_step)))

    const_pop_del = np.delete(const_pop, 1, 0)

    uniq_const, contts = np.unique(const_pop_del, axis=1, return_counts=True)

    wdw_req = np.arange(time_wind_step)
    # print(uniq_const[1,:].shape)
    wdw_req = np.tile(wdw_req, n_targets)

    if np.all(wdw_req == uniq_const[1, :]):
        print('OK! Coverage time requirement satisfied!')
    else:
        print('ERROR! Coverage time requirement NOT satisfied')


def const_stats(cov_pop, const_vec, rev_time, n_step, n_targets, step_l):
    mask = np.isin(cov_pop[1, :], const_vec)
    const_pop = cov_pop[:, mask]

    max_rt_v = np.empty(n_targets)
    min_rt_v = np.empty(n_targets)
    mean_rt_v = np.empty(n_targets)

    for t in np.arange(n_targets):
        times_tar = const_pop[2, const_pop[0, :] == t]
        times_tar = np.sort(times_tar, axis=0)
        un_time = np.unique(times_tar)

        delta_time = np.diff(un_time)
        max_delta = np.amax(delta_time)
        max_rev_time = max_delta * step_l / 60  # [min]
        max_rt_v[t] = max_rev_time

        print('Target', t + 1)
        print('Max r.t.', np.around(max_rev_time[0], 2), ' mins')

        min_delta = np.amin(delta_time[delta_time != 1])  # diregard continuous access
        min_rev_time = min_delta * step_l  # [sec]
        min_rt_v[t] = min_rev_time

        print('Min r.t.', np.around(min_rev_time[0], 2), ' secs')

        mean_delta = np.mean(delta_time[delta_time != 1])  # diregard continuous access
        mean_rev_time = mean_delta * step_l / 60  # [min]
        mean_rt_v[t] = mean_rev_time

        print('Mean r.t.', np.around(mean_rev_time[0], 2), ' mins \n')

    global_max = np.amax(max_rt_v)
    global_min = np.amin(min_rt_v)
    global_mean = np.mean(mean_rt_v)

    print('Global max r.t.', np.around(global_max, 2), ' mins')
    print('Global min r.t.', np.around(global_min, 2), ' secs')
    print('Global mean r.t.', np.around(global_mean, 2), ' mins \n')

    return 0


def constellation_function():
    # Inputs
    # date of simulation
    year = 2022
    month = 4
    day = 1

    e = 0  # eccentricity

    N_p = 13  #
    N_d = 1
    tau = N_p / N_d

    wpo = 0 * deg  # argument of perigee for all orbits

    TAo = 0 * deg  # true Anomaly
    n_periods = N_p  # number of periods for which ground track is to be plotted

    n_OM = 2 * 360
    Wo = 0 * deg  # RAAN Right Ascension of the Ascending Node
    W_o = Wo + np.array([np.linspace(0, 2 * np.pi, n_OM)]).T
    # W_o = Wo + np.array([np.linspace(4/3*np.pi, 5/3*np.pi, n_OM)]).T

    n_incl = 10
    incl_min = 60
    incl_max = 70
    incl = np.linspace(incl_min, incl_max, n_incl) * deg

    t_length = 60  # [sec / step] duration of each timestep

    f_acr = 31 * deg
    f_alo = 16 * deg

    rev_time = 120  # [min] revisit time
    print('Revisit time: ', rev_time, '[min] \n')

    lon_t, lat_t = read_targets()

    r_t = latlon2car(lat_t, lon_t, Re)

    a = tau2a(tau, e, incl, J2, Re, we, mu)

    (r, v, step_l, W, wp, TA, theta, MM, times) = kep2car(a, e, incl, W_o, wpo, TAo, n_periods, t_length)

    print('Orbits propagated \n')

    n_step = r.shape[3]  # Number timesteps
    # max_pairs = filt_anV3(a, r, v, r_t, f_acr, f_alo, rev_time, n_step)

    # ind_W, ind_i, a_b = best_pair(a, Re, W_o, incl, max_pairs)

    # r_best = r[:, ind_W, ind_i, :]

    # plot_exp_track(r_best, lat_t, lon_t, N_p)

    n_step = r.shape[3]  # Number timesteps
    div_par = 3
    N_s = n_step / div_par  # Number satellites poplutation
    N_t = lat_t.shape[0]  # Number targets

    # Om_pop, M_pop = pair_pop(N_s, N_p, N_d, W_o[ind_W, 0], 0, incl[ind_i], a[0,ind_i], e, n_step)

    # r_v, v_v = sat_pop(a_b, e, incl[ind_i], Om_pop, wpo, M_pop, n_periods, step_l[:, ind_i], n_step)

    cov_steps = cov_steps_fun(a, r, v, r_t, f_acr, f_alo, rev_time, n_step)

    Om_index_vec, incl_index_vec = set_cover_prblm_targets(cov_steps, r_t, rev_time, n_step)
    print('Subconstellations:', Om_index_vec, incl_index_vec)

    a_b, r_const, v_const, Om_pop, M_pop = const_rv_vec(Om_index_vec, incl_index_vec, N_s, N_p, N_d, W_o, incl, a, e,
                                                        n_step, step_l, times, wpo, n_periods, year, month, day)

    cov_pop = filt_pop(a_b, r_const, v_const, r_t, f_acr, f_alo)

    cont_time_check(cov_pop, n_step)

    incl_opt = incl_index_vec[0]
    # step_l[:, incl_opt]

    n_targets = lon_t.shape[0]
    # uni_mat = cov_uni(rev_time, step_l[0,ind_i], N_d, n_step, n_targets)
    uni_mat = cov_uni(rev_time, step_l[0, incl_opt], N_d, n_step, n_targets)

    set_mat = cov_set(cov_pop, uni_mat, rev_time, n_step, N_d)

    print('cov_pop shape: ', cov_pop.shape)
    print(cov_pop)
    print('\nset_mat shape: ', set_mat.shape)
    print(set_mat)
    print('--')

    const_vec, pry_vec = cov_probl(set_mat, N_t)
    print(pry_vec)
    # print(const_vec)

    Om_const, M_const = const_info(const_vec, Om_pop, M_pop, n_step)

    best_incl = incl_index_vec[0]  # they are the same anyways
    const_mat = const_mat_fun(a_b, e, incl[best_incl], Om_const, M_const, wpo)

    const_verfy(cov_pop, const_vec, rev_time, n_step, n_targets, N_d)

    incl_opt = incl_index_vec[0]
    const_stats(cov_pop, const_vec, rev_time, n_step, n_targets, step_l[:, incl_opt])


def asymmetric():
    # --- INPUTS ---
    # date of simulation
    year = 2022
    month = 4
    day = 1

    e = 0     # eccentricity

    N_p = 13  # number of orbit periods. Correlated with repeating ground track orbit theory
    N_d = 1   # number of Grenwich nodal periods. Correlated with repeating ground track orbit theory
    tau = N_p / N_d   # directly correlated with altitude

    wpo = 0 * deg  # (initial) argument of perigee

    TAo = 0 * deg  # (initial) true Anomaly
    n_periods = N_p  # number of periods for which ground track is to be plotted

    n_OM = 2 * 360  # number of RAAN values in RAAN range
    Wo = 0 * deg  # (initial) RAAN Right Ascension of the Ascending Node
    W_o = Wo + np.array([np.linspace(0, 2 * np.pi, n_OM)]).T   # create RAAN range for orbit population

    n_incl = 10     # number of inclination values in inclination range
    incl_min = 60   # minimum inclination value range
    incl_max = 70   # maximum inclination value range
    incl = np.linspace(incl_min, incl_max, n_incl) * deg  # create inclination range for orbit population

    t_length = 60  # [sec / step] duration of each timestep

    f_acr = 31 * deg  # across track angle
    f_alo = 16 * deg  # along track angle

    rev_time = 120  # [min] revisit time
    print('Revisit time: ', rev_time, '[min] \n')

    # ---- End Inputs ----

    lon_t, lat_t = read_targets()     # import of targets

    r_t = latlon2car(lat_t, lon_t, Re)  # transform targets from lat/lon to x,y,z coordinates

    a = tau2a(tau, e, incl, J2, Re, we, mu)   # calculate altitude based on tau, e, inclination

    (r, v, step_l, W, wp, TA, theta, MM, times) = kep2car(a, e, incl, W_o, wpo, TAo, n_periods, t_length, year, month, day)   # propagation of orbits

    print('Orbits propagated \n')

    n_step = r.shape[3]  # Number timesteps
    div_par = 3
    N_s = n_step / div_par  # Number satellites poplutation
    N_t = lat_t.shape[0]  # Number targets

    cov_steps = cov_steps_fun(a, r, v, r_t, f_acr, f_alo, rev_time, n_step)     # create access profiles

    Om_index_vec, incl_index_vec = set_cover_prblm_targets(cov_steps, r_t, rev_time, n_step)   # choose how many and which subconstellations
    print('Subconstellations:', Om_index_vec, incl_index_vec)

    a_b, r_const, v_const, Om_pop, M_pop = const_rv_vec(Om_index_vec, incl_index_vec, N_s, N_p, N_d, W_o, incl, a, e,
                                                        n_step, step_l, times, wpo, n_periods, year, month, day)    # altitude, position and velocity vector for constellation

    cov_pop = filt_pop(a_b, r_const, v_const, r_t, f_acr, f_alo)  # analysis of coverage

    cont_time_check(cov_pop, n_step)   # check if coverage is doing ok

    incl_opt = incl_index_vec[0]    # optimal inclination

    n_targets = lon_t.shape[0]     # number of targets
    uni_mat = cov_uni(rev_time, step_l[0, incl_opt], N_d, n_step, n_targets)

    set_mat = cov_set(cov_pop, uni_mat, rev_time, n_step, N_d)

    print('cov_pop shape: ', cov_pop.shape)
    print(cov_pop)
    print('\nset_mat shape: ', set_mat.shape)
    print(set_mat)
    print('--')

    const_vec, pry_vec = cov_probl(set_mat, N_t)    # solves set cover problem (satellite distribution)
    print(pry_vec)

    Om_const, M_const = const_info(const_vec, Om_pop, M_pop, n_step)   # stores RAAN and mean anomaly information for constellation

    best_incl = incl_index_vec[0]  # they are the same anyways
    const_mat = const_mat_fun(a_b, e, incl[best_incl], Om_const, M_const, wpo)    # constellation matrix

    const_verfy(cov_pop, const_vec, rev_time, n_step, n_targets, N_d)  # verification if revisit time is guaranteed

    incl_opt = incl_index_vec[0]
    const_stats(cov_pop, const_vec, rev_time, n_step, n_targets, step_l[:, incl_opt])   # covering analytics

    return const_mat




# Grab Currrent Time Before Running the Code
start = time.time()

constellation_matrix = asymmetric()

# Grab Currrent Time After Running the Code
end = time.time()
# Subtract Start Time from The End Time
total_time = end - start
print("\n execution time: " + str(total_time) + " seconds")

cProfile.run('asymmetric()')
