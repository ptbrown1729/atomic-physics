# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:12:06 2014

@author: Peter
"""

from __future__ import print_function
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numerov as num
import wigner
#import rydberg as ryd

def get_psi(atom, qnumbers, steps, solver='linear'):
    """
    Solve for U(r)=r * R(r) using the numerov algorithm for integration
    :param atom:
    :param qnumbers:
    :param steps:
    :param solver:
    :return:
    """
    psi1 = ryd.rydberg(atom, qnumbers)

    solve_psi1 = num.numerov(psi1.numerov_fun, psi1.alphac ** (1 / 3), 2 * qnumbers[0] * (qnumbers[0] + 15))
    if solver == "linear":
        rad_psi1 = solve_psi1.back_numerov(steps)
    elif solver == "log":
        rad_psi1 = solve_psi1.log_back_numerov(steps)
    elif solver == "quarter":
        rad_psi1 = solve_psi1.quarter_back_numerov(steps)
    else:
        raise ValueError("Solver type must be linear, log, quarter. See numerov.py for detailed information.")
    # interpolate wavefunction
    rad_psi1_interp = interp1d(solve_psi1.rpts, rad_psi1)

    # get normalization
    def norm1(r):
        return rad_psi1_interp(r) ** 2

    # set range
    ri = solve_psi1.rpts[0]
    rf = solve_psi1.rpts[-1]

    N1 = quad(norm1, ri, rf)[0]

    def func(r):
        return rad_psi1_interp(r) / math.sqrt(N1)

    return func

def rad_int(psi1, psi2, ri, rf):
    """
    Computes radial integral, <n1 l1 |er| n2 l2>.
    :param psi1: Radial wavefunction for the first state, U(r) = r * R(r)
    :param psi2: Radial wavefunction for the second state
    :param ri: minimum radius for integration
    :param rf: maximum radius for integration
    :return: integral
    """

    def func(r):
        return psi1(r) * psi2(r) * r

    # set bounds of integration and integrate
    return quad(func, ri, rf)[0]

# get full matrix element
def get_coupled_mat_element(qnumbers1, qnumbers2, irr_tensor_index, radial_matrix_elem=1.):
    """
    Compute <((L'S')J'I')F' m_F' | u(1, q) | ((LS)JI)F m_F >
    this should be identical with
    <((LS)JI)F m_F | u(1, -q) | ((L'S')J'I')F' m_F' >*

    Reference for the reduction to <L' || u(1) || L>: http://www.steck.us_interspecies/alkalidata/rubidium87numbers.1.6.pdf
    Also see, but there are some typos, in particular in the second wigner 6j symbol: www.physics.ncsu.edu/jet/techdocs/pcf/PropertiesOfLi.pdf
    Reduction of <L' || u(1) || L> to radial matrix element: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.68.054701
    Note that there are two different conventions for the reduced matrix elements, so must be careful. Steck and Gehm
     use different conventions

    :param qnumbers1: [L', S', J', I', F', mf']
    :param qnumbers2: [L, S, J, I, F, mf]
    :param irr_tensor_index: 1, 0, or -1
    :param radial_matrix_elem:
    :return: matrix_element
    """

    lp = qnumbers1[0]
    sp = qnumbers1[1]
    jp = qnumbers1[2]
    ip = qnumbers1[3]
    fp = qnumbers1[4]
    mfp = qnumbers1[5]

    l = qnumbers2[0]
    s = qnumbers2[1]
    j = qnumbers2[2]
    i = qnumbers2[3]
    f = qnumbers2[4]
    mf = qnumbers2[5]

    # mat_elem = (-1) ** (fp - mfp) * np.sqrt(2*f + 1) * wigner.Wigner3j(fp, 1, f, -mfp, q, mfp) * red_mat_elem_F
    # two conventions actually...one included the sqrt(2*f + 1) in the above, and one does not
    # or equivalently = <F mf | F' 1 mf' q> * red_mat_elem_F

    q = irr_tensor_index
    prefactor = (-1) ** (fp - mfp + jp + i + f + 1 + lp + s + j + 1 + lp) * \
                math.sqrt((2*fp + 1) * (2*f + 1) * (2*jp + 1) * (2*j + 1) * (2*lp + 1) * (2*l + 1))
    delta = (i == ip) * (s == sp)
    wigner_symbols = wigner.Wigner3j(fp, 1, f, -mfp, q, mf) * wigner.Wigner6j(jp, i, fp, f, 1, j) * \
                     wigner.Wigner6j(lp, s, jp, j, 1, l) * wigner.Wigner3j(l, 1, lp, 0, 0, 0)

    return prefactor * delta * wigner_symbols * radial_matrix_elem

def get_mat_elem_from_redF(qnumbers1, qnumbers2, irr_tensor_index, reduced_mat_elem_F=1., convention='cg'):
    """
    Compute <((L'S')J'I')F' m_F' | u(1, q) | ((LS)JI)F m_F > from the irreducible matrix element <F' || u(1) || F>
    :param qnumbers1: [L', S', J', I', F', mf']
    :param qnumbers2: [L, S, J, I, F, mf]
    :param irr_tensor_index:
    :param reduced_mat_elem_F: <F' || u(1) || F>
    :param convention: "cg" or "wigner"
    :return:
    """
    lp = qnumbers1[0]
    sp = qnumbers1[1]
    jp = qnumbers1[2]
    ip = qnumbers1[3]
    fp = qnumbers1[4]
    mfp = qnumbers1[5]

    l = qnumbers2[0]
    s = qnumbers2[1]
    j = qnumbers2[2]
    i = qnumbers2[3]
    f = qnumbers2[4]
    mf = qnumbers2[5]

    q = irr_tensor_index

    if convention == "cg":
        factor = math.sqrt(2 * fp + 1)
    elif convention == "wigner":
        factor = 1.
    else:
        raise error()

    # these two expressions are equivalent
    matrix_el = get_mat_el_wigner_eckart([fp, mfp], [1, q], [f, mf], reduced_mat_elem=reduced_mat_elem_F, convention=convention)
    # matrix_el = (-1) ** (f - 1 + mfp) * factor * wigner.Wigner3j(f, 1, fp, mf, q, -mfp) * reduced_mat_elem_F
    # matrix_el = (-1) ** (fp - mfp) * factor * wigner.Wigner3j(fp, 1, f, -mfp, q, mf) * reduced_mat_elem_F
    return matrix_el

def get_mat_elem_from_redJ(qnumbers1, qnumbers2, irr_tensor_index, reduced_mat_elem_J=1., convention='cg'):
    """
    Compute <((L'S')J'I')F' m_F' | u(1, q) | ((LS)JI)F m_F > from the irreducible matrix element <J'' || u(1) || J>

    :param qnumbers1: [L', S', J', I', F', mf']
    :param qnumbers2: [L, S, J, I, F, mf]
    :param irr_tensor_index:
    :param reduced_mat_elem_J: <J' || u(1) || J>
    :param convention: "cg" or "wigner"
    :return:
    """
    lp = qnumbers1[0]
    sp = qnumbers1[1]
    jp = qnumbers1[2]
    ip = qnumbers1[3]
    fp = qnumbers1[4]
    mfp = qnumbers1[5]

    l = qnumbers2[0]
    s = qnumbers2[1]
    j = qnumbers2[2]
    i = qnumbers2[3]
    f = qnumbers2[4]
    mf = qnumbers2[5]

    reduced_mat_elem_F = get_reduced_mat_elem_F([jp, ip, fp], [j, i, f], reduced_mat_elem_J, convention=convention)
    return get_mat_elem_from_redF(qnumbers1, qnumbers2, irr_tensor_index, reduced_mat_elem_F, convention=convention)

def get_mat_elem_from_redL(qnumbers1, qnumbers2, irr_tensor_index, reduced_mat_elem_L=1., convention='cg'):
    """
        Compute <((L'S')J'I')F' m_F' | u(1, q) | ((LS)JI)F m_F > from the reduced matrix element <L' || u(1) || L>

        :param qnumbers1: [L', S', J', I', F', mf']
        :param qnumbers2: [L, S, J, I, F, mf]
        :param irr_tensor_index: 1, 0, or -1
        :param reduced_mat_elem_L:
        :param convention: "cg" or "wigner"
        :return: matrix_element
        """

    lp = qnumbers1[0]
    sp = qnumbers1[1]
    jp = qnumbers1[2]
    ip = qnumbers1[3]
    fp = qnumbers1[4]
    mfp = qnumbers1[5]

    l = qnumbers2[0]
    s = qnumbers2[1]
    j = qnumbers2[2]
    i = qnumbers2[3]
    f = qnumbers2[4]
    mf = qnumbers2[5]

    reduced_mat_elem_J = get_reduced_mat_elem_J([lp, sp, jp], [l, s, j], reduced_mat_elem_L, convention=convention)
    return get_mat_elem_from_redJ(qnumbers1, qnumbers2, irr_tensor_index, reduced_mat_elem_J, convention=convention)

# get reduced matrix elements
def get_reduced_mat_elem_F(qnumbers1, qnumbers2, reduced_mat_elem_J=1., convention='cg'):
    """
    Compute the reduced matrix element <(J'I')F' || u(1) || (JI)F >.

    This matrix element can be defined using two different conventions, which we will term "cg" for Clebsch-Gordan and
    "wigner" below. Note: the Gehm datasheet uses the "wigner" convention and the Steck datasheets use the "cg" convention.

    #######################
    # "cg" convention
    #######################
    Using the "cg" convention this is defined by:
    <(J'I') F' mf' | u(1,q) | (JI) F mf> = <F mf | F' mf' 1 q> <(J'I')F' || u(1) || (JI)F >
    we can write this numerous different ways using symmetries of the 3j coefficients.
    = (-1) ** (F - 1 + mf') * sqrt(2F' + 1) * wigner3j(F, 1, F', mf, q, -mf') * <(J'I')F' || u(1) || (JI)F >_cg
    = (-1) ** (F' - mf')    * sqrt(2F' + 1) * wigner3j(F', 1, F, -mf', q, mf) * <(J'I')F' || u(1) || (JI)F >_cg

    We can write the reduced F matrix element in terms of the reduced J matrix element
    <(J'I')F' || u(1) || (JI)F >_cg =     (-1) ** (F + J' + 1 + I) * sqrt( (2F + 1) (2J' + 1) ) * wigner6J(J', I, F', F, 1, J)

    #######################
    # "wigner" convention
    #######################
    Using the "wigner" convention we have the following definition:
    <(J'I') F' mf' | u(1,q) | (JI) F mf> =
    = (-1) ** (F - 1 + mf') * wigner3j(F, 1, F', mf, q, -mf') * <(J'I')F' || u(1) || (JI)F >_wigner

    Or in terms of the reduced J matrix element:
    <(J'I')F' || u(1) || (JI)F >_wigner = (-1) ** (F + J' + 1 + I) * sqrt( (2F + 1) (2F' + 1) ) * wigner6J(J', I, F', F, 1, J)

    #######################
    # relationship between the two conventions
    #######################
    <(J'I')F' || u(1) || (JI)F >_wigner = sqrt(2F + 1) * <(J'I')F' || u(1) || (JI)F >_cg

    :param qnumbers1: [J', I', F']
    :param qnumbers2: [J, I, F]
    :param reduced_mat_elem_J:
    :param convention: either "cg" for Clebsch-Gordan because the relationship for the full matrix element is related
    to the product of a CG coefficient and the reduced matrix element. Or, "wigner". See above documentation for more
    information
    :return:
    """

    Jp = qnumbers1[0]
    Ip = qnumbers1[1]
    Fp = qnumbers1[2]

    J = qnumbers2[0]
    I = qnumbers2[1]
    F = qnumbers2[2]

    if convention == "cg":
        factor = math.sqrt((2 * F + 1) * (2 * Jp + 1))
    elif convention == "wigner":
        factor = math.sqrt((2 * Fp + 1) * (2 * F + 1))
    else:
        raise error()
    return (Ip == I) * (-1) ** (Jp + I + F + 1) * factor * wigner.Wigner6j(Jp, I, Fp, F, 1, J) * reduced_mat_elem_J

def get_reduced_mat_elem_J(qnumbers1, qnumbers2, reduced_mat_elem_L=1., convention='cg'):
    """
    Compute the reduced matrix element <(L'S')J' || u(1) || (LS)J >. See the documentation for the function
    get_reduced_mat_elem_F() for more information

    :param qnumbers1: [L', S', J']
    :param qnumbers2: [L, S, J]
    :param reduced_mat_elem_L:
    :param convention: "cg" or "wigner"
    :return:
    """
    # Compute <(L'S')J' || u(1) || (LS)J >
    lp = qnumbers1[0]
    sp = qnumbers1[1]
    jp = qnumbers1[2]

    l = qnumbers2[0]
    s = qnumbers2[1]
    j = qnumbers2[2]

    # exactly the same form as <(J'I')F' || u(1) || (JI) F>
    # with substitution L -> J, S -> I, F -> J
    return get_reduced_mat_elem_F([lp, sp, jp], [l, s, j], reduced_mat_elem_L, convention=convention)

def get_reduced_mat_elem_L(qnumbers1, qnumbers2, radial_mat_elem=1., convention='cg'):
    """
    Compute the reduced matrix element <L'|| u(1) || L>.
    Typically using L' is the ground-state and L is the excited state.

    :param qnumbers1: [L']
    :param qnumbers2: [L]
    :param radial_mat_elem:
    :param convention: "cg" or "wigner"
    :return:
    """

    Lp = qnumbers1[0]
    L = qnumbers2[0]

    if convention == "cg":
        factor = math.sqrt((2 * L + 1))
    elif convention == "wigner":
        factor = math.sqrt((2 * L + 1) * (2 * Lp + 1))
    else:
        raise error()

    # TODO: so far no good way to check the numerical prefactor here
    # But see page 420 of "2017-Atomic-Physics" Appendix J. Irreducible Tensor Operators is useful
    return (-1) ** Lp * factor * wigner.Wigner3j(L, 1, Lp, 0, 0, 0) * radial_mat_elem

# misc reduction functions
def get_mat_el_wigner_eckart(qnumbers1, qnumbers2, qnumbers3, reduced_mat_elem=1., convention='cg'):
    """
    Compute <J' mj' | T(k, q) | J mj> using the Wigner-Eckart theorem.

    Using the 'cg' convention, we have
    <J' mj' | T(k, q) | J mj> = (-1)**(J - k - mj') * wigner3j(J, k, J', mj, q, -mj') * sqrt(2*J' + 1) * <J' || T(k) || J>
                              = <J mj k q | J' mj'> * <J' || T(k) || J>

    Using the 'wigner' convention, we have
    <J' mj' | T(k, q) | J mj> = (-1)**(J - k - mj') * wigner3j(J, k, J', mj, q, -mj') * <J' || T(k) || J>

    The value of the reduced matrix elements depends on the choice of convention, and we see from the above that
    <J' || T(k) || J>_wigner = sqrt(2*J' + 1) * <J' || T(k) || J>_cg

    :param qnumbers1:
    :param qnumbers2:
    :param qnumbers3:
    :param reduced_matrix_el:
    :param convention:
    :return:
    """

    Jp = qnumbers1[0]
    mjp = qnumbers1[1]

    k = qnumbers2[0]
    q = qnumbers2[1]

    J = qnumbers3[0]
    mj = qnumbers3[1]

    if convention == "cg":
        factor = math.sqrt(2 * Jp + 1)
    elif convention == "wigner":
        factor = 1.
    else:
        raise error()

    return (-1) ** (J - k + mjp) * wigner.Wigner3j(J, k, Jp, mj, q, -mjp) * factor * reduced_mat_elem

def get_spherical_harm_triple_product(qnumbers1, qnumbers2, qnumbers3):
    """
    Compute the product of three spherical harmonics, which we can write as
     <L' ml' | Y^q_k(r) | L ml> = \int dr Y^{ml'}_{L'}(r) * Y^q_k(r) * Y^{ml}_L(r)

    :param qnumbers1:
    :param qnumbers2:
    :param qnumbers3:
    :return:
    """

    Lp = qnumbers1[0]
    mlp = qnumbers1[1]

    k = qnumbers2[0]
    q = qnumbers2[1]

    L = qnumbers3[0]
    ml = qnumbers3[1]

    return (-1) ** mlp * np.sqrt( (2*Lp + 1) * (2*k + 1) * (2*L + 1) / (4*np.pi) ) * wigner.wigner3j(Lp, k, L, 0, 0, 0) * wigner.wigner3j(Lp, k, L, -mlp, q, ml)

# get matrix elements between lines for all states in hyperfine basis
def get_all_coupled_mat_elements(qnumbers_1, qnumbers_2, reduced_mat_elem=1, mode="L", convention='cg'):
    """
    compute matrix elements in the coupled basis for each possible polarization. <F' mf' | u(1, q) | F, mf>, where the
    primed variables correspond to qnumbers_1 and the unprimed correspond to qnumbers_2
    :param qnumbers_1: [ll', jj', ii']
    :param qnumbers_2: [ll, jj, ii]
    :param reduced_mat_elem: either <J' || u(1) || J>, <L' || u(1) || L>, or R_{n'l'}{nl} depending on the value of the mode argument.
    :param mode: If "J" compute the matrix elements from the value of <J' || u(1) || J>. If "L", compute the matrix
    elements from the value <L' || u(1) || L>. If "R" compute the matrix elements from the value of
    R_{n'l''}{nl}
    :param convention: either "cg" or "wigner"
    :return:
    [mat_el_sigma_plus, mat_el_sigma_minus, mat_el_pi]: list of matrix elements matrices
    polarizations: list of polarizations
    """

    ss = 0.5
    ll_1 = qnumbers_1[0]
    jj_1 = qnumbers_1[1]
    ii_1 = qnumbers_1[2]

    ll_2 = qnumbers_2[0]
    jj_2 = qnumbers_2[1]
    ii_2 = qnumbers_2[2]

    # get bases
    coupled_basis_p = get_coupled_basis([ll_1, jj_1, ii_1])
    n1 = coupled_basis_p.shape[0]

    coupled_basis = get_coupled_basis([ll_2, jj_2, ii_2])
    n2 = coupled_basis.shape[0]

    mat_sigp = np.zeros((n1, n2))
    mat_sigm = np.zeros((n1, n2))
    mat_pi = np.zeros((n1, n2))

    for ii in range(0, n1):
        ff_1 = coupled_basis_p[ii, 0]
        mf_1 = coupled_basis_p[ii, 1]

        # d1 line
        for jj in range(0, n2):
            ff_2 = coupled_basis[jj, 0]
            mf_2 = coupled_basis[jj, 1]

            # [L, S, J, I, F, mf]
            qnums_full_1 = [ll_1, ss, jj_1, ii_1, ff_1, mf_1]
            qnums_full_2 = [ll_2, ss, jj_2, ii_2, ff_2, mf_2]

            if mode == "L":
                mat_sigp[ii, jj] = get_mat_elem_from_redL(qnums_full_1, qnums_full_2, 1, reduced_mat_elem, convention=convention)
                mat_sigm[ii, jj] = get_mat_elem_from_redL(qnums_full_1, qnums_full_2, -1, reduced_mat_elem, convention=convention)
                mat_pi[ii, jj] = get_mat_elem_from_redL(qnums_full_1, qnums_full_2, 0, reduced_mat_elem, convention=convention)
            elif mode == "J":
                mat_sigp[ii, jj] = get_mat_elem_from_redJ(qnums_full_1, qnums_full_2, 1, reduced_mat_elem, convention=convention)
                mat_sigm[ii, jj] = get_mat_elem_from_redJ(qnums_full_1, qnums_full_2, -1, reduced_mat_elem, convention=convention)
                mat_pi[ii, jj] = get_mat_elem_from_redJ(qnums_full_1, qnums_full_2, 0, reduced_mat_elem, convention=convention)
            elif mode == "R":
                mat_sigp[ii, jj] = get_coupled_mat_element(qnums_full_1, qnums_full_2, 1, radial_matrix_elem=1)
                mat_sigm[ii, jj] = get_coupled_mat_element(qnums_full_1, qnums_full_2, -1, radial_matrix_elem=1)
                mat_pi[ii, jj] = get_coupled_mat_element(qnums_full_1, qnums_full_2, 0, radial_matrix_elem=1)
            else:
                raise error()

    return [mat_sigp, mat_sigm, mat_pi], [1, -1, 0], coupled_basis_p, coupled_basis

# calculate decay rates
def get_semiclassical_decay_rate(lambda_o):
    e = 1.602e-19
    m_e = 9.10938356e-31
    c = 299792458  # m/s
    w = 2 * np.pi * c / lambda_o
    epsilon_o = 8.854187817e-12  # farads/meter

    gamma = e ** 2 * w ** 2 / ( 6 * np.pi * epsilon_o * m_e * c ** 3)

    return gamma

def get_decay_rate(matrix_el, lambda_o):
    """
    Compute the decay rate (einstein A-coefficient). For a two levels atom, set J=0. For a real atom, we can interpret
    this decay rate as the rate of decay for a single |F mf> state into any |F' mf'> state in a certain |J LS> manifold.
    :param matrix_el: in coulombs * meters
    :param lambda_o: in meters
    :return: decay rate in hertz
    """
    c = 299792458  # m/s
    w = 2 * np.pi * c / lambda_o
    epsilon_o = 8.854187817e-12 # farads/meter
    h = 6.62607004e-34
    hbar = h / (2 * np.pi)
    gamma = w ** 3 / (3 * np.pi * epsilon_o * c ** 3) * np.abs(matrix_el) ** 2 / hbar
    return gamma

def get_decay_rate_Dline(lambda_o, J, Jp, reduced_mat_elem_J, convention='cg'):
    """
    Decay rate Gamma = (2*pi) * f in radians for any Zeeman sublevel, |(LSJI) F mf> to a single J' level and all
    values of F' and mf'
    :param lambda_o:
    :param J:
    :param Jp:
    :param reduced_mat_elem_J:
    :param convention:
    :return:
    """
    c = 299792458  # m/s
    w = 2 * np.pi * c / lambda_o
    epsilon_o = 8.854187817e-12  # farads/meter
    h = 6.62607004e-34
    hbar = h / (2 * np.pi)
    if convention == "wigner":
        factor = 1 / (2 * J + 1)
    elif convention == "cg":
        factor = (2 * Jp + 1) / (2 * J + 1)
    else:
        raise error()

    gamma = w ** 3 / (3 * np.pi * epsilon_o * c ** 3) * factor *  np.abs(reduced_mat_elem_J) ** 2 / hbar
    return gamma

def gamma2matrixel_Dline(lambda_o, J, gamma, convention='cg'):
    """
    Compute the absolute value of the reduced matrix elements <J'=0.5 || u(1) || J> and <L'=0 || u(1) || L = 1>
    for either the D1 or D2 line.

    We have the relationship
    hbar * Gamma = omega^3/(3*pi*epsilon_o*c^3) \sum_g,q |<g| u(1,q) | e>|^2
    and \sum_g,q |<g| u(1,q) | e>|^2 = \sum_{F' mf', q} |<(L'S'J'I') F' mf' | u(1,q) | (LSJI) F mf>|^2
    = 1/(2J+1) * |<J' || u(1) || J>|^2 in the "wigner" convention
    = (2J' + 1)/(2J + 1) * |<J' || u(1) || J>|^2 in the "cg" convention

    :param lambda_o: wavelength of the D1 or D2 transition
    :param J: J of the excited state. J=0.5 for the D1 line and J=1.5 for the D2 line
    :param gamma: decay rate in radians, i.e. gamma = (2pi) * decay_freq
    :param convention: "cg" or "wigner"
    :return:
    |<J' || u(1) || J>|, |<L' || u(1) || L>|
    """
    c = 299792458  # m/s
    w = 2 * np.pi * c / lambda_o
    epsilon_o = 8.854187817e-12  # farads/meter
    h = 6.62607004e-34
    hbar = h / (2 * np.pi)

    # for transition to nS_{1/2} ground-state
    Jp = 0.5

    if convention == "cg":
        factor = (2 * Jp + 1) / (2 * J + 1)
    elif convention == "wigner":
        factor = 1 / (2 * J + 1)
    else:
        raise error()

    # <J'=0.5 || u(1) || J>
    reduced_mat_elem_J = np.sqrt( (hbar * gamma) /  (w ** 3 / (3 * np.pi * epsilon_o * c ** 3) * factor) )

    # <L'=0 || u(1) || L=1>
    # [L, S, J]
    j_helper = get_reduced_mat_elem_J([0, 0.5, Jp], [1, 0.5, J], reduced_mat_elem_L=1, convention=convention)
    reduced_mat_elem_L = reduced_mat_elem_J / j_helper

    return reduced_mat_elem_J, reduced_mat_elem_L

# calculate light shifts and decay rates in off-resonant dipole trap
def get_lightshift_semiclass(lambda_o, gamma, lambda_laser, intensity):
    """
    Get lightshift and scattering rate for an atom in an off-resonant dipole trap using the semiclassical oscillator.

    This gives a lightshift of dE = 3 * pi * c^2 /(2*wo^3) * Gamma (1 / Delta + 1 / (wo + wl) ) * I.
    or large detunings we must also keep the second term. For small detunings, the first term dominates.

    The scattering rate is Gamma_eff/(2pi) = 3 * pi * c^2/(2*wo^3) * (w/wo)^3 * (1 / Delta + 1 / (wo + wl))^2 * I / h

    Note that for small detunings we have
    dE = 3 * pi * c^2 /(2*wo^3) * Gamma / Delta * I = matrix_el^2 / Delta * |E|^2
    Gamma_eff/(2pi) = 3 * pi * c^2/(2*wo^3) * (w/wo)^3 * (Gamma / Delta)^2 * I
    which is the second order perturbation theory result!

    see arxiv:physics/9902072v1
    :param lambda_o: atomic transition wavelength in meters
    :param gamma: excited state decay rate in radians (i.e. gamma = (2pi) * frq)
    :param lambda_laser: laser wavelength in meters
    :param intensity: intensity in watts per meter^2
    :return:
    lightshift in joules
    gamma in hertz
    """
    c =  299792458 # m/s
    h = 6.62607004e-34 # J * s
    hbar = h / (2 * np.pi)
    wo = 2 * np.pi * c / lambda_o
    ws = 2 * np.pi * c /lambda_laser
    lightshift = -3 * np.pi * c ** 2 / (2 * wo ** 3) * gamma * (np.divide(1, wo - ws) + np.divide(1, wo + ws)) * intensity
    gamma = 3 * np.pi * c ** 2 / (2 * wo ** 3) * np.divide(ws, wo) ** 3 * gamma ** 2 * (np.divide(1, wo - ws) + np.divide(1, wo + ws))**2 * intensity / hbar
    return lightshift, gamma

def get_all_lightshift_semiclass(lambda_d1, lambda_d2, gamma, lambda_laser, intensity, gf, F):
    """
    Calculate lightshift of ground state hyperfine levels in a single F manifold accounting only for the D line.

    Note that this expression is good in the approximation that the detuning Delta is small compared to the size of the
    absolute frequency. It should agree with the result from get_lightshift_semiclass when the detuning is large compared
    with the fine structure splitting. in that case, Delta_d1 ~ Delta_d2 and the expression reduceds to the other function
    to order Delta_fs / Delta

    :param lambda_d1:
    :param lambda_d2:
    :param gamma:
    :param lambda_laser:
    :param intensity:
    :param irr_tensor_index:
    :param gf:
    :param F:
    :return:
    """
    fbasis = get_spin_basis(F)

    c = 299792458  # m/s
    wd1 = 2 * np.pi * c / lambda_d1
    wd2 = 2 * np.pi * c / lambda_d2
    wo = (2 * wd2 + wd1) / 3.
    #wo = 0.5 * (wd1 + wd2)
    ws = 2 * np.pi * c / lambda_laser

    if np.abs( (wo - ws) / (wo + ws) ) > 0.1:
        raise Warning('The detuning is not small compared to twice the frequency. Accuracy may suffer.')

    shift_fn = lambda q: np.pi * c ** 2 / (2 * wo ** 3) * gamma * ( (2 + q * gf * fbasis[:, 1]) / (ws - wd1) + (1 - q * gf * fbasis[:, 1]) / (ws - wd2) ) * intensity
    shift = np.concatenate((shift_fn(1)[:, None], shift_fn(-1)[:, None], shift_fn(0)[:, None]), axis=1)
    return shift, [1, -1, 0], fbasis

def get_lightshift_Dline(b_fields, lambda_d1, lambda_d2, lambda_laser, intensity,
                         I, params_g, params_d1, params_d2, reduced_matrixel_l, convention='cg'):
    """
    Compute lightshifts for each Zeeman sublevel of the ground state accounting only for the D1 and D2 lines using
    second order perturbation theory.

    :param b_fields:
    :param lambda_d1: d1 transition wavelength in meters
    :param lambda_d2: d2 transition wavelength in meters
    :param lambda_laser: laser wavelength in meters
    :param intensity: laser intensity in W/m^2
    :param I: nuclear spin quantum numbers
    :param params_g: hyperfine coupling and g-factors of the nS_{1/2} ground state [Ahf, gs, gl, gi]
    :param params_d1: hyperfine coupling and g-factors of the nP_{1/2} excited state [Ahf, gs, gl, gi]
    :param params_d2: hyperfine coupling and g-factors of the nP_{3/2} excited state [Ahf, gs, gl, gi]
    :param reduced_matrixel_l: < L' || u(1) || L > in either the "cg" or "wigner" convention
    :param convention: either "cg" or "wigner". See e.g. get_reduced_matrixel_F() for a more detailed explanation
    :return:
    light_shifts: a list of three arrays. Each array represents lightshifts from a certain type of polarized light.
    polarization_indices: List of polarization angular momentum corresponding to the light_shifts list
    """

    h = 6.62607004e-34
    hbar = h / (2 * np.pi)
    c = 299792458  # m/s
    epsilon_o = 8.854187817e-12  # farads/meter
    # energies in MHz
    ed1 = c / lambda_d1 / 1e6
    ed2 = c / lambda_d2 / 1e6
    # expression for J = 0.5 -> 1.5, since these two states are assymetrically split about the line
    ed = (2. * ed2 + ed1) / 3.
    elaser = c / lambda_laser / 1e6

    b_fields = np.asarray(b_fields)
    if b_fields.ndim == 0:
        b_fields = b_fields[None]

    # [L, J, I]
    qnumbers_g = [0, 0.5, I]
    qnumbers_d1 = [1, 0.5, I]
    qnumbers_d2 = [1, 1.5, I]

    # ##################################
    # compute matrix elements in the coupled basis for each possible polarization
    # <F mf | u(1, q) | F', mf'>
    # Ground states are primes, excited states are not.
    # ##################################
    matel_d1, _, _, _ = get_all_coupled_mat_elements(qnumbers_d1, qnumbers_g, reduced_matrixel_l, mode="L",
                                                     convention=convention)
    mat_sigp_d1 = matel_d1[0]
    mat_sigm_d1 = matel_d1[1]
    mat_pi_d1 = matel_d1[2]

    matel_d2, _, _, _ = get_all_coupled_mat_elements(qnumbers_d2, qnumbers_g, reduced_matrixel_l, mode="L",
                                                     convention=convention)
    mat_sigp_d2 = matel_d2[0]
    mat_sigm_d2 = matel_d2[1]
    mat_pi_d2 = matel_d2[2]

    # ##################################
    # Compute eigentstates at each magnetic field
    # These are matrices <F mf | i>, with |i> the eigenstate. i.e. the columns of our matrix are eigenvectors
    # ##################################
    # ground state, eigenstates
    energies_g, eigvects_uncoupled_g, eigvects_coupled_g = breit_rabi(b_fields, params_g, qnumbers_g)
    coupled_states_g = get_coupled_basis(qnumbers_g)
    ng = eigvects_coupled_g.shape[0]

    # J = 0.5 excited state
    energies_d1, eigvects_uncoupled_d1, eigvects_coupled_d1 = breit_rabi(b_fields, params_d1, qnumbers_d1)
    coupled_states_d1 = get_coupled_basis(qnumbers_d1)
    nd1 = eigvects_coupled_d1.shape[0]

    # J = 1.5 excited state
    energies_d2, eigvects_uncoupled_d2, eigvects_coupled_d2 = breit_rabi(b_fields, params_d2, qnumbers_d2)
    coupled_states_d2 = get_coupled_basis(qnumbers_d2)
    nd2 = eigvects_coupled_d2.shape[0]

    # ##################################
    # now get matrix elements in eigenbasis for each magnetic field
    # M[ii, jj] = <i_e | F, mf> <F mf | u(1, q) | F' mf'> <F' mf' | j_g>
    # ##################################

    matrix_elems_sigp_d1 = np.zeros((nd1, ng, b_fields.size))
    matrix_elems_sigm_d1 = np.zeros((nd1, ng, b_fields.size))
    matrix_elems_pi_d1 = np.zeros((nd1, ng, b_fields.size))
    #
    matrix_elems_sigp_d2 = np.zeros((nd2, ng, b_fields.size))
    matrix_elems_sigm_d2 = np.zeros((nd2, ng, b_fields.size))
    matrix_elems_pi_d2 = np.zeros((nd2, ng, b_fields.size))

    for ii in range(0, b_fields.size):
        # d1
        matrix_elems_sigp_d1[:, :, ii] = eigvects_coupled_d1[:, :, ii].conj().transpose().dot(
            mat_sigm_d1).dot(eigvects_coupled_g[:, :, ii])

        matrix_elems_sigm_d1[:, :, ii] = eigvects_coupled_d1[:, :, ii].conj().transpose().dot(
            mat_sigp_d1).dot(eigvects_coupled_g[:, :, ii])

        matrix_elems_pi_d1[:, :, ii] = eigvects_coupled_d1[:, :, ii].conj().transpose().dot(
            mat_pi_d1).dot(eigvects_coupled_g[:, :, ii])

        # d2
        matrix_elems_sigp_d2[:, :, ii] = eigvects_coupled_d2[:, :, ii].conj().transpose().dot(
            mat_sigm_d2).dot(eigvects_coupled_g[:, :, ii])

        matrix_elems_sigm_d2[:, :, ii] = eigvects_coupled_d2[:, :, ii].conj().transpose().dot(
            mat_sigp_d2).dot(eigvects_coupled_g[:, :, ii])

        matrix_elems_pi_d2[:, :, ii] = eigvects_coupled_d2[:, :, ii].conj().transpose().dot(
            mat_pi_d2).dot(eigvects_coupled_g[:, :, ii])

    # ##################################
    # compute lightshift from second order perturbation theory
    # ##################################
    # E/h = |<e| H |g>|^2 * |E|^2 / (h * delta) / h
    #     = |<e| H |g>|^2 * (I/2 e_o c) / (h * delta) / h

    # energies differences, take advantage of broadcasting
    e_diffs_d1 = ed1 + (energies_d1[:, None, :] - energies_g[None, :, :])
    e_diffs_d2 = ed2 + (energies_d2[:, None, :] - energies_g[None, :, :])

    light_shifts_sp = np.sum(np.abs(matrix_elems_sigp_d1) ** 2 / (elaser - e_diffs_d1), axis=0) + \
                      np.sum(np.abs(matrix_elems_sigp_d2) ** 2 / (elaser - e_diffs_d2), axis=0)
    light_shifts_sm = np.sum(np.abs(matrix_elems_sigm_d1) ** 2 / (elaser - e_diffs_d1), axis=0) + \
                      np.sum(np.abs(matrix_elems_sigm_d2) ** 2 / (elaser - e_diffs_d2), axis=0)
    light_shifts_pi = np.sum(np.abs(matrix_elems_pi_d1) ** 2 / (elaser - e_diffs_d1), axis=0) + \
                      np.sum(np.abs(matrix_elems_pi_d2) ** 2 / (elaser - e_diffs_d2), axis=0)

    # convert |<>|^2 * |E|^2 joules to MHz
    light_shifts = np.concatenate((light_shifts_sp[:, :, None], light_shifts_sm[:, :, None],
                                   light_shifts_pi[:, :, None]), axis=2) / (h * 1e6) ** 2

    # convert from |E|^2 to I
    light_shifts = intensity / (2 * epsilon_o * c) * light_shifts

    # ##################################
    # calculate scattering rate
    # ##################################
    # Gamma_eff = Gamma * |Omega|^2 / (4 * Delta^2) \prop matrix_el^4
    #
    gamma_sp = (np.sum(np.abs(matrix_elems_sigp_d1) ** 4 / (elaser - e_diffs_d1) ** 2, axis=0) +
                np.sum(np.abs(matrix_elems_sigp_d2) ** 4 / (elaser - e_diffs_d2) ** 2, axis=0))
    gamma_sm = (np.sum(np.abs(matrix_elems_sigm_d1) ** 4 / (elaser - e_diffs_d1) ** 2, axis=0) +
                np.sum(np.abs(matrix_elems_sigm_d2) ** 4 / (elaser - e_diffs_d2) ** 2, axis=0))
    gamma_pi = (np.sum(np.abs(matrix_elems_pi_d1) ** 4 / (elaser - e_diffs_d1) ** 2, axis=0) +
                np.sum(np.abs(matrix_elems_pi_d2) ** 4 / (elaser - e_diffs_d2) ** 2, axis=0))

    # scale factor
    gamma = (gamma_sp + gamma_sm + gamma_pi) * (2 * np.pi * ed) ** 3 / (
                6 * np.pi * epsilon_o ** 2 * c ** 4 * hbar ** 2) * intensity
    # convert Deltas from MHz to Hz
    gamma = gamma / 1e6 ** 2 / hbar

    return light_shifts, [1, -1, 0], gamma

# calculate branching ratios
def get_branching_ratios(b_fields, atomic_params_g, qnumbers_g, atomic_params_e, qnumbers_e):
    """
    Given two atomic transitions, print the branching ratios between the excited and ground states.
    :param b_fields: magnetic field to compute branching ratios at
    :param atomic_params_g: list, [A, gs, gl, gi]. A is the coefficient of the A*I.dot(J) term in the atomic
    hamiltonian. gs is the spin gyromagnetic moment, gl is the angular gyromagnetic moment, gi is the nuclear
    gyromagnetic moment.
    :param qnumbers_g: [L, J, I], where L is the orbital angular momentum quantum number, J = L + S, and I is the
    nuclear angular momentum quantum number
    :param atomic_params_e:
    :param qnumbers_e:
    :return: matrix_elem_list, tensor_indices
    """

    b_fields = np.asarray(b_fields)
    if b_fields.ndim == 0:
        b_fields = b_fields[None]

    # ground state, eigenstates
    nstates_g = int((2 * qnumbers_g[1] + 1) * (2 * qnumbers_g[2] + 1))
    energies_g, eigvects_uncoupled_g, eigvects_coupled_g = breit_rabi(b_fields, atomic_params_g, qnumbers_g)
    coupled_states_g = get_coupled_basis(qnumbers_g)

    # excited state, eigenstates
    nstates_e = int((2 * qnumbers_e[1] + 1) * (2 * qnumbers_e[2] + 1))
    energies_e, eigvects_uncoupled_e, eigvects_coupled_e = breit_rabi(b_fields, atomic_params_e, qnumbers_e)
    coupled_states_e = get_coupled_basis(qnumbers_e)

    # ##################################
    # compute matrix elements in the coupled basis for each possible polarization
    # <F' mf' | u(1, q) | F mf>
    # Ground states are primes, excited states are not.
    # ##################################
    matel, _, _, _ = get_all_coupled_mat_elements(qnumbers_g, qnumbers_e, reduced_mat_elem=1, mode="R", convention="cg")

    mat_sigp = matel[0]
    mat_sigm = matel[1]
    mat_pi = matel[2]

    # now get matrix elements in eigenbasis
    matrix_elems_sigp = np.zeros((nstates_g, nstates_e, b_fields.size))
    matrix_elems_sigm = np.zeros((nstates_g, nstates_e, b_fields.size))
    matrix_elems_pi = np.zeros((nstates_g, nstates_e, b_fields.size))

    for ii in range(0, b_fields.size):
        matrix_elems_sigp[:, :, ii] = eigvects_coupled_g[:, :, ii].conj().transpose().dot(mat_sigp).dot(eigvects_coupled_e[:, :, ii])
        matrix_elems_sigm[:, :, ii] = eigvects_coupled_g[:, :, ii].conj().transpose().dot(mat_sigm).dot(eigvects_coupled_e[:, :, ii])
        matrix_elems_pi[:, :, ii]   = eigvects_coupled_g[:, :, ii].conj().transpose().dot(mat_pi).dot(eigvects_coupled_e[:, :, ii])

    # absolute value of matrix elements
    abs_sigp = matrix_elems_sigp * matrix_elems_sigp.conj()
    abs_sigm = matrix_elems_sigm * matrix_elems_sigm.conj()
    abs_pi   = matrix_elems_pi * matrix_elems_pi.conj()
    abs = abs_sigp + abs_sigm + abs_pi

    # these should all be the same. i.e., the net decay rate out of hyperfine states in the same manifold are the same
    # norms_sigp = np.sum(abs_sigp, 0)
    # norms_sigm = np.sum(abs_sigm, 0)
    # norms_pi   = np.sum(abs_pi, 0)
    norms_all = np.sum(abs, 0)
    # all norms should be same
    norm = norms_all[0, 0]

    # TODO: make work for multiple Bs at once
    abs_norm = abs / norm
    abs_sigp_norm = abs_sigp / norm
    abs_sigm_norm = abs_sigm / norm
    abs_pi_norm = abs_pi / norm

    # ###################################
    # Create description strings
    # ###################################
    str_list = []

    for kk in range(0, b_fields.size):
        str = '****************************************************'
        str = str + '********** B = %0.2fG **********\n' % b_fields[kk]
        str = str + '********** all transitions **********\n'
        for jj in range(0, nstates_e):
            str = str +  "|%2d> -> " % (jj + 1)
            for ii in range(0, nstates_g):
                if abs_norm[ii, jj, kk] > 1e-3:
                    str = str + "%+0.3f |%2d> " % (abs_norm[ii, jj, kk], ii + 1)
            str = str + '\n'
        str = str + '\n'

        str = str + '********** sigma plus transitions **********\n'
        for jj in range(0, nstates_e):
            str = str +  "|%2d> -> " % (jj + 1)
            for ii in range(0, nstates_g):
                if abs_sigp_norm[ii, jj, kk] > 1e-3:
                    str = str + "%+0.3f |%2d> " % (abs_sigp_norm[ii, jj, kk], ii + 1)
            str = str + '\n'

        str = str + '********** sigma minus transitions **********\n'
        for jj in range(0, nstates_e):
            str = str + "|%2d> -> " % (jj + 1)
            for ii in range(0, nstates_g):
                if abs_sigm_norm[ii, jj, kk] > 1e-3:
                    str = str + "%+0.3f |%2d> " % (abs_sigm_norm[ii, jj, kk], ii + 1)
            str = str + "\n"

        str = str + '********** pi transitions **********\n'
        for jj in range(0, nstates_e):
            str = str + "|%2d> -> " % (jj + 1)
            for ii in range(0, nstates_g):
                if abs_pi_norm[ii, jj, kk] > 1e-3:
                    str = str + "%+0.3f |%2d> " % (abs_pi_norm[ii, jj, kk], ii + 1)
            str = str + "\n"

        str_list.append(str)

    # we can also ask if we start in a given ground state, excite with sigma plus light to the excited states, and
    # look at the decay, what what states do the atoms end up in?
    # ns = np.power(norms_sigp, -1)
    # ns[np.isinf(ns)] = 0.
    # n_mat = np.diag(ns)

    # expect rows to add to one, but they don't. Need to think about this.
    # a = np.round((abs / norms[0] ).dot( (abs_sigp.dot(n_mat)).transpose() ), 6)

    matrix_elem_list = [matrix_elems_sigp, matrix_elems_sigm, matrix_elems_pi]
    branching_ratio_list = [abs_sigp_norm, abs_sigm_norm, abs_pi_norm]
    tensor_indices = [1, -1, 0]

    return matrix_elem_list, branching_ratio_list, tensor_indices, str_list

# magnetic fields
def get_gj(gfactors, qnumbers):
    """
    Get approximate gj from gs, gl, and quantum numbers. This is useful when the L.S coupling is strong compared to
    the Zeeman shift term.
    :param gfactors: [gs, gl]
    :param qnumbers: [ss, ll, jj]
    :return:
    """
    gs = gfactors[0]
    gl = gfactors[1]
    ss = qnumbers[0]
    ll = qnumbers[1]
    jj = qnumbers[2]

    gj = gl * (jj * (jj + 1) - ss * (ss + 1) + ll * (ll + 1)) / (2 * jj * (jj + 1)) + \
         gs * (jj * (jj + 1) + ss * (ss + 1) - ll * (ll + 1)) / (2 * jj * (jj + 1))
    return gj

def get_gf(gfactors, qnumbers):
    """
    Get approximate gf from gj, gi, and quantum numbers. This is useful when the I.J (hyperfine) coupling is strong compared to
    the Zeeman shift term.

    :param gfactors: [gj, gi]
    :param qnumbers: [jj, ii, ff]
    :return gf:
    """
    gj = gfactors[0]
    gi = gfactors[1]
    jj = qnumbers[0]
    ii = qnumbers[1]
    ff = qnumbers[2]
    # same expression as gj because always combine two spins in same way
    return get_gj([gi, gj], [ii, jj, ff])

def breit_rabi(b_fields, atomic_params, qnumbers):
    """
    Solve Hamiltonian H = A * I.J + mub * ( gj * J + gi * I) * B. Return energies and eigenstates in both the
    "coupled" i.e. |F JI mf> and "uncoupled" i.e. |J mj, I mi> bases.

    Note that gj is calculated from gs and gl.

    :param b_fields: magnetic field
    :param atomic_params: =  [A, gs, gl, gi]
    :param qnumbers: = [LL, JJ, II]
    :return eig_energies: array of eigenenergies of size nbasis x nbfields
    :return eigvects_uncoupled: array of eigenvectors in uncoupled basis of size nbasis x nbasis x nbfields.
    :return eigvects_coupled: array of eigenvectors in coupled basis of size nbasis x nbasis x nbfields.
    """

    b_fields = np.asarray(b_fields)
    if b_fields.ndim == 0:
        b_fields = b_fields[None]

    # TODO: should I split off the basis calculations into another function?
    # TODO: how to ensure that B = 0 states are sorted correctly?

    A, gs, gl, gi = atomic_params

    ll, jj, ii = qnumbers
    ss = 0.5

    # construct desired bases
    nstates = int((2*ii + 1) * (2*jj + 1))

    coupled_states = get_coupled_basis(qnumbers)
    uncoupled_states = get_uncoupled_basis(qnumbers)
    # basis change matrix, from clebsch gordon coefficients
    # |ff mf> = \sum <(j mj) (i mi) | ff mf> * |(j mj) (i mi)>
    uncoupled2coupled_mat = np.zeros((nstates, nstates))
    for aa, cstate in enumerate(coupled_states):
        for bb, ustate in enumerate(uncoupled_states):
            mi = ustate[1]
            mj = ustate[0]
            ff = cstate[0]
            mf = cstate[1]
            uncoupled2coupled_mat[aa, bb] = wigner.clebsch_gordan(jj, ii, ff, mj, mi, mf)

    # hyperfine part of hamiltonian
    h_hyperfine_coupled = A * np.diag(0.5 * (coupled_states[:, 0] * (coupled_states[:, 0] + 1) - jj * (jj + 1) - ii * (ii + 1)))
    h_hyperfine_uncoupled = uncoupled2coupled_mat.conj().transpose().dot(h_hyperfine_coupled.dot(uncoupled2coupled_mat))

    # magnetic field part of hamiltonian
    mu_bohr = 9.27400968e-24 / 6.62607004e-34 / 1e4 / 1e6  # MHz/ G
    gj = get_gj([gs, gl], [ss, ll, jj])
    h_zeeman_unc = mu_bohr * (gj * np.diag(uncoupled_states[:, 0]) + gi * np.diag(uncoupled_states[:, 1]) )

    # full hamiltonian and diagonalization
    eig_energies = np.zeros((nstates, b_fields.size))
    eigvects_uncoupled = np.zeros((nstates, nstates, b_fields.size))
    eigvects_coupled = np.zeros((nstates, nstates, b_fields.size))
    for ii in range(0, b_fields.size):
        hamiltonian_uncoupled = h_hyperfine_uncoupled + b_fields[ii] * h_zeeman_unc
        eig_energies[:, ii], eigvects_uncoupled[:, :, ii] = np.linalg.eigh(hamiltonian_uncoupled)
        # transformation to coupled basis. That this is the appropriate transformation is clear if you regard the columns
        # of eigvects_uncoupled as vectors
        eigvects_coupled[:, :, ii] = uncoupled2coupled_mat.dot(eigvects_uncoupled[:, :, ii])

    return eig_energies, eigvects_uncoupled, eigvects_coupled

def get_magnetic_moments(b_field, atomic_params, qnumbers, tol=1e-4, max_iters=100):
    """
    Get magnetic moment of state at given field

    :param b_field: In gauss
    :param atomic_params: [A, gs, gl, gi]
    :param qnumbers: [L, J, I]
    :return:
    """
    #tol = 1e-4
    #max_iters = 100

    # get number of states
    jj = qnumbers[1]
    ii = qnumbers[2]
    # construct desired bases
    nstates = int((2 * ii + 1) * (2 * jj + 1))

    dBs = np.zeros(max_iters)
    diffs = np.zeros(max_iters)
    mag_mom = np.zeros((nstates, max_iters))

    dBs[0] = 1
    diffs[0] = np.inf

    # iteratively determine moments until reach fine enough precision
    for ii in range(0, max_iters):

        # get moment
        es,  _, _ = breit_rabi([b_field + dBs[ii], b_field - dBs[ii]], atomic_params, qnumbers)
        mag_mom[:, ii] = ( es[:, 0] - es[:, 1] ) / (2 * dBs[ii])

        # get difference
        if ii > 0:
            diffs[ii] = np.max(np.abs(mag_mom[:, ii] - mag_mom[:, ii-1]))

        # prepare for next loop
        dBs[ii + 1] = dBs[ii] / 3

        if diffs[ii] < tol or ii == max_iters:
            break

    # trim to number of iterations. Could think about returning these as a check in the future
    dBs = dBs[:ii+1]
    diffs = diffs[:ii+1]
    mag_mom = mag_mom[:, :ii+1]

    return mag_mom[:, -1]

def breit_rabi_SLI(b_field, atomic_params, gfactors, qnumbers):
    """
    Solve Hamiltonian H = Als * L.S + Ali * L.I + Asi * S.I + mub * ( gl * L + gs * s + gi * I) * B. Return energies and eigenstates in both the "coupled"
    (i.e. |F JI mf>) and "uncoupled" (i.e. |J mj, I mi>> bases.
    Note that gj is calculated from gs and gl.
    :param b_field: magnetic field
    :param atomic_params: =  [Als, Ali, Asi]
    :param gfactors: =  [gs, gl, gi]
    :param qnumbers: = [L, I]
    :return eig_energies: array of eigen energies of size nbasis x nbfields
    :return eigvects_uncoupled: array of eigenvectors in uncoupled basis of size nbasis x nbasis x nbfields.
    :return eigvects_coupled: array of eigenvectors in coupled basis of size nbasis x nbasis x nbfields.
    """

    # TODO: FINISH!
    b_field = np.asarray(b_field)
    if b_field.ndim == 0:
        b_field = b_field[None]

    Als = atomic_params[0]
    Ali = atomic_params[1]
    Asi = atomic_params[2]

    gs = gfactors[0]
    gl = gfactors[1]
    gi = gfactors[2]

    ll = qnumbers[0]
    ii = qnumbers[1]
    ss = 0.5

    # construct desired bases
    nstates = int((2 * ll + 1) * (2 * ii + 1) * (2 * ss + 1))

    coupled_states = get_coupled_basis(qnumbers)
    uncoupled_states = get_uncoupled_basis(qnumbers)
    # basis change matrix, from clebsch gordon coefficients
    # |ff mf> = \sum <(j mj) (i mi) | ff mf> * |(j mj) (i mi)>
    uncoupled2coupled_mat = np.zeros((nstates, nstates))
    for aa, cstate in enumerate(coupled_states):
        for bb, ustate in enumerate(uncoupled_states):
            mi = ustate[1]
            mj = ustate[0]
            ff = cstate[0]
            mf = cstate[1]
            uncoupled2coupled_mat[aa, bb] = wigner.clebsch_gordan(jj, ii, ff, mj, mi, mf)

    # hyperfine part of hamiltonian
    h_hyperfine_coupled = A * np.diag(
        0.5 * (coupled_states[:, 0] * (coupled_states[:, 0] + 1) - jj * (jj + 1) - ii * (ii + 1)))
    h_hyperfine_uncoupled = uncoupled2coupled_mat.conj().transpose().dot(h_hyperfine_coupled.dot(uncoupled2coupled_mat))

    # magnetic field part of hamiltonian
    mu_bohr = 9.27400968e-24 / 6.62607004e-34 / 1e4 / 1e6  # MHz/ G
    h_zeeman_unc = mu_bohr * gj * np.diag(uncoupled_states[:, 0]) + b_field * mu_bohr * gi * np.diag(
        uncoupled_states[:, 1])

    # full hamiltonian and diagonalization
    eig_energies = np.zeros((nstates, b_field.size))
    eigvects_uncoupled = np.zeros((nstates, nstates, b_field.size))
    eigvects_coupled = np.zeros((nstates, nstates, b_field.size))
    for ii in range(0, b_field.size):
        hamiltonian_uncoupled = h_hyperfine_uncoupled + b_field[ii] * h_zeeman_unc
        eig_energies[:, ii], eigvects_uncoupled[:, :, ii] = np.linalg.eigh(hamiltonian_uncoupled)
        # transformation to coupled basis. That this is the appropriate transformation is clear if you regard the columns
        # of eigvects_uncoupled as vectors
        eigvects_coupled[:, :, ii] = uncoupled2coupled_mat.dot(eigvects_uncoupled[:, :, ii])

    eig_energies = np.squeeze(eig_energies)
    eigvects_uncoupled = np.squeeze(eigvects_uncoupled)
    eigvects_coupled = np.squeeze(eigvects_coupled)

    return eig_energies, eigvects_uncoupled, eigvects_coupled

# get basis states
def get_spin_basis(spin):
    """
    Return spin basis as array of size nstates x 2, where the first column gives the total spin number and the second
    column gives the magnetic quantum number ms.
    :param spin:
    :return:
    """
    nstates = int( 2 * spin + 1)

    ms = np.arange(-spin, spin + 1, 1)
    basis = np.concatenate((np.ones((nstates, 1)) * spin, ms[:, None]), axis=1)

    return basis

def get_coupled_basis(qnumbers):
    """
    Return a two column matrix representing coupled atomic basis states, i.e. states with quantum numbers
    |L J F mf>. The first column of the coupled_state_matrix are the F's, and the second column are the mf's
    :param qnumbers: [L, J, I]
    :return: coupled_state_matrix
    """
    ll = qnumbers[0]
    jj = qnumbers[1]
    ii = qnumbers[2]
    ss = 0.5

    nstates = int((2 * ii + 1) * (2 * jj + 1))

    # coupled states list (i.e. eigenstates of I.J interaction)
    # nstates 2 x matrix
    # first column ff's, second column mf's
    ff_allowed = np.arange(np.abs(ii - jj), ii + jj + 1)
    for index, ff in enumerate(ff_allowed):
        mfs = np.arange(-ff, ff + 1)
        cstates = np.concatenate((ff * np.ones((mfs.size, 1)), np.reshape(mfs, [mfs.size, 1])), 1)
        if index == 0:
            coupled_states = cstates
        else:
            coupled_states = np.concatenate((coupled_states, cstates), 0)
    return coupled_states

def get_uncoupled_basis(qnumbers):
    """
    Return a two column matrix representing uncoupled atomic basis states, i.e. states with quantum numbers
    |J mj, I mi>. The first column of the uncoupled_state_matrix are the mj's, and the second column are the mi's
    :param qnumbers: [L, J, I]
    :return: uncoupled_state_matrix
    """
    ll = qnumbers[0]
    jj = qnumbers[1]
    ii = qnumbers[2]
    ss = 0.5
    # uncoupled states (i.e. eigenstates of B.J + B.I interaction)
    # nstates x 2 matrix
    # first column mj's, second column mi's
    mjs = np.arange(-jj, jj + 1)
    mis = np.arange(-ii, ii + 1)
    for index, mj in enumerate(mjs):
        ustates = np.concatenate((mj * np.ones((mis.size, 1)), np.reshape(mis, [mis.size, 1])), 1)
        if index == 0:
            uncoupled_states = ustates
        else:
            uncoupled_states = np.concatenate((uncoupled_states, ustates), 0)

    return uncoupled_states

# display functions
def print_coupled_eigenstates(coupled_eigstates, coupled_basis):
    """
    Print eigenstates in terms of the uncoupled basis states in a human readable form
    :param coupled_eigstates: Each column of this matrix is an eigenstate expressed in the coupled basis
    :param coupled_basis: 2 x nstates matrix, where the first column gives the F quantum number, and the second gives
    the mF quantum number for the basis state corresponding to that row
    :return:
    """
    nstates = coupled_eigstates.shape[0]
    coupled_eigstates = np.round(coupled_eigstates, 6)

    for jj in range(0, nstates):
        print("|%2d> = " % (jj + 1), end='')
        for ii in range(0, nstates):
            if coupled_eigstates[ii, jj] > 0:
                print("%+0.2f |F=%+0.1f, mf=%+0.1f> " % (coupled_eigstates[ii, jj], coupled_basis[ii, 0], coupled_basis[ii, 1]), end='')
        print('', end='\n')
    print('', end='\n')

def print_uncoupled_eigenstates(uncoupled_eigstates, uncoupled_basis):
    """
    Print eigenstates in terms of the uncoupled basis states in a human readable form
    :param uncoupled_eigstates: Each column of this matrix is an eigenstate expressed in the uncoupled basis
    :param uncoupled_basis: 2 x nstates matrix, where the first column gives the mj quantum number, and the second gives
    the mj quantum number for the basis state corresponding to that row
    :return:
    """
    nstates = uncoupled_eigstates.shape[0]
    uncoupled_eigstates = np.round(uncoupled_eigstates, 6)

    for jj in range(0, nstates):
        print("|%2d> = " % (jj + 1), end='')
        for ii in range(0, nstates):
            if uncoupled_eigstates[ii, jj] > 0:
                print("%+0.2f |mj=%+0.1f, mi=%+0.1f> " % (
                uncoupled_eigstates[ii, jj], uncoupled_basis[ii, 0], uncoupled_basis[ii, 1]), end='')
        print('', end='\n')
    print('', end='\n')

def plot_states_vs_field(b_fields, energies, uncoupled_basis, uncoupled_eigvects, coupled_basis, coupled_eigvects, desc_str='', plot_log=0):
    """
    Plot energies versus magnetic field and eigenstate overlap with the uncoupled basis (|J mj; I mi>) and the coupled
    or hyperfine basis (|F J I mf>). Generate the energies and overlap matrices using the function breit_rabi
    :param b_fields:
    :param energies:
    :param uncoupled_basis:
    :param uncoupled_eigvects:
    :param coupled_basis:
    :param coupled_eigvects:
    :param desc_str:
    :return:
    """
    nstates = coupled_basis.shape[0]

    # overlap of eigenstates with coupled/uncoupled states
    coupled_fracs = coupled_eigvects * coupled_eigvects.conj()
    uncoupled_fracs = uncoupled_eigvects * uncoupled_eigvects.conj()

    # coupled states ordered by
    coupled_states_ordered = np.zeros((nstates, 2))
    uncoupled_states_ordered = np.zeros((nstates, 2))
    for ii in range(0, nstates):
        index = np.argsort(coupled_fracs[:, ii, 0])[-1]
        coupled_states_ordered[ii, :] = coupled_basis[index, :]

        index = np.argsort(uncoupled_fracs[:, ii, -1])[-1]
        uncoupled_states_ordered[ii, :] = uncoupled_basis[index, :]

    # #############################
    # plot energies vs. field
    # #############################
    energies_fig = plt.figure()
    plt.plot(b_fields, energies.transpose())

    plt.xlabel('Magnetic field (G)')
    plt.ylabel('Energy (MHz)')
    plt.title('%s, %s, hyperfine substate energies versus magnetic field' % (desc_str, desc_str))
    plt.grid()
    plt.xlim([0, 1.2 * np.max(b_fields)])

    # annotations labelling states
    ax = energies_fig.axes[0]
    for ii in range(0, nstates):
        # find coupled state near zero field
        str = '|F = %+0.1f mf = %+0.1f>' % (coupled_states_ordered[ii, 0], coupled_states_ordered[ii, 1])
        ax.annotate(str, xy=(b_fields[5], energies[ii, 5]), xytext=(b_fields[5], energies[ii, 5]),
                    xycoords='data', textcoords='data')

        # find uncoupled state near high field
        str = '|mj = %+0.1f mi = %+0.1f>' % (uncoupled_states_ordered[ii, 0], uncoupled_states_ordered[ii, 1])
        ax.annotate(str, xy=(np.max(b_fields), energies[ii, -1]),
                    xytext=(np.max(b_fields), energies[ii, -1]),
                    xycoords='data', textcoords='data')

    # #############################
    # plot eigenstate overlaps with uncoupled basis states
    # #############################
    uncoupled_fig = plt.figure()
    ncols = np.ceil(np.sqrt(nstates))
    nrows = np.ceil(nstates / ncols)

    leg = []
    for ii in range(0, nstates):

        plt.subplot(nrows, ncols, ii + 1)
        if plot_log:
            plt.semilogy(b_fields, uncoupled_fracs[:, ii, :].transpose())
            plt.ylim([10**-5, 1.1])
        else:
            plt.plot(b_fields, uncoupled_fracs[:, ii, :].transpose())

        plt.xlabel('Magnetic field (G)')
        plt.ylabel('')
        plt.title('State %d' % (ii + 1))
        plt.grid()
        leg.append('mj = %0.1f, mi = %0.1f' % (uncoupled_basis[ii, 0], uncoupled_basis[ii, 1]))

        if ii == (nstates - 1):
            leg = plt.legend(leg)
            # leg.draggable()
    plt.suptitle('%s, eigenstate overlaps with uncoupled basis states' % (desc_str))

    # #############################
    # plot eigenstate overlaps with coupled basis states
    # #############################
    coupled_fig = plt.figure()
    ncols = np.ceil(np.sqrt(nstates))
    nrows = np.ceil(nstates / ncols)

    leg = []
    for ii in range(0, nstates):
        plt.subplot(nrows, ncols, ii + 1)
        if plot_log:
            plt.semilogy(b_fields, coupled_fracs[:, ii, :].transpose())
            plt.ylim([10**-5, 1.1])
        else:
            plt.plot(b_fields, coupled_fracs[:, ii, :].transpose())

        plt.xlabel('Magnetic field (G)')
        plt.ylabel('')
        plt.title('State %d' % (ii + 1))
        plt.grid()

        leg.append('F = %0.1f, mf = %0.1f' % (coupled_basis[ii, 0], coupled_basis[ii, 1]))
        if ii == (nstates - 1):
            leg = plt.legend(leg)
            # leg.draggable()
    plt.suptitle('%s, eigenstate overlaps with coupled basis states' % (desc_str))

    return energies_fig, uncoupled_fig, coupled_fig

def plot_branching_ratios(b_fields, branching_ratios, desc_str='', plot_log=0):

    nstates = branching_ratios.shape[0]
    nstates_e = branching_ratios.shape[1]

    # plot branching ratios
    branching_fig = plt.figure()
    ncols = np.ceil(np.sqrt(nstates_e))
    nrows = np.ceil(nstates_e / ncols)

    leg = []
    for ii in range(0, nstates):
        leg.append('|%2d>' % (ii + 1))
    for ii in range(0, nstates_e):
        plt.subplot(nrows, ncols, ii + 1)
        if plot_log:
            plt.semilogy(b_fields, branching_ratios[:, ii, :].transpose())
            plt.ylim([10**-5, 1.1])
        else:
            plt.plot(b_fields, branching_ratios[:, ii, :].transpose())

        if ii >= nrows * ncols - ncols:
            plt.xlabel('Magnetic field (G)')
        plt.ylabel('')
        plt.title('State %d' % (ii + 1))
        plt.grid()

        if ii == (nstates_e - 1):
            leg = plt.legend(leg)
            # leg.draggable()
    plt.suptitle('%s branching ratios' % (desc_str))

    return branching_fig

if __name__ == "__main__":
    pass