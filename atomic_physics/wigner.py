"""
Code to calculate Clebsch-Gordan coefficients, Wigner3j symbols, and Wigner6j symbols
Most functions here were derived from Matlab code written by others
"""

from scipy.special import factorial
import numpy as np


def clebsch_gordan(j1, j2, j, m1, m2, m, print_warnings=False):
    """
    Compute Clebsch gordan coefficient <(j1 m1) (j2 m2) | j m >

    :param j1:
    :param j2:
    :param j:
    :param m1:
    :param m2:
    :param m:
    :param print_warnings:
    :return:
    """
    return (-1)**(j1-j2+m) * np.sqrt(2*j + 1) * wigner3j(j1, j2, j, m1, m2, -m, print_warnings)


def wigner3j(j1, j2, j3, m1, m2, m3, print_warnings=False):
    """
    Wigner3j.m by David Terr, Raytheon, 6-17-04
    Compute the Wigner 3j symbol using the Racah formula. See e.g.
    Wigner 3j-Symbol entry of Eric Weinstein's Mathworld:
    http://mathworld.wolfram.com/Wigner3j-Symbol.html
     / j1 j2 j3 \
     |          |
     \ m1 m2 m3 /

    :param j1:
    :param j2:
    :param j3:
    :param m1:
    :param m2:
    :param m3:
    :param print_warnings:
    :return:
    """

    # Error checking
    if ((2*j1 != np.floor(2*j1)) |
        (2*j2 != np.floor(2*j2)) |
        (2*j3 != np.floor(2*j3)) |
        (2*m1 != np.floor(2*m1)) |
        (2*m2 != np.floor(2*m2)) |
        (2*m3 != np.floor(2*m3))):
        print('All arguments must be integers or half-integers.')
        return -1

    # Additional check if the sum of the second row equals zero
    if m1+m2+m3 != 0:
        if print_warnings:
            print('3j-Symbol unphysical')
        return 0

    if j1 - m1 != np.floor(j1 - m1):
        if print_warnings:
            print('2*j1 and 2*m1 must have the same parity')
        return 0
    
    if j2 - m2 != np.floor(j2 - m2):
        if print_warnings:
            print('2*j2 and 2*m2 must have the same parity')
        return 0

    if (j3 - m3) != np.floor(j3 - m3):
        if print_warnings:
            print('2*j3 and 2*m3 must have the same parity')
        return 0
    
    if (j3 > j1 + j2) | (j3 < np.abs(j1 - j2)):
        if print_warnings:
            print('j3 is out of bounds.')
        return 0

    if abs(m1) > j1:
        if print_warnings:
            print('m1 is out of bounds.')
        return 0

    if abs(m2) > j2:
        if print_warnings:
            print('m2 is out of bounds.')
        return 0 

    if abs(m3) > j3:
        if print_warnings:
            print('m3 is out of bounds.')
        return 0

    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2

    tmin = max(0, max(t1, t2))
    tmax = min(t3, min(t4, t5))
    tvec = np.arange(tmin, tmax+1, 1)

    wigner = 0

    for t in tvec:
        wigner += (-1)**t / (factorial(t) * factorial(t - t1) *
                             factorial(t - t2) * factorial(t3 - t) *
                             factorial(t4 - t) * factorial(t5 - t))

    w3j = wigner * (-1) ** (j1-j2-m3) * np.sqrt(factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) *
                                                factorial(-j1 + j2 + j3) / factorial(j1 + j2 + j3 + 1) *
                                                factorial(j1 + m1) * factorial(j1 - m1) *
                                                factorial(j2 + m2) * factorial(j2 - m2) *
                                                factorial(j3 + m3) * factorial(j3 - m3))

    return w3j


def wigner6j(j1, j2, j3, J1, J2, J3, print_warnings=False):
    """
    Calculating the Wigner6j-Symbols using the Racah-Formula
    Author: Ulrich Krohn
    Date: 13th November 2009

    Based upon Wigner3j.m from David Terr, Raytheon
    Reference: http://mathworld.wolfram.com/Wigner6j-Symbol.html
    / j1 j2 j3 \
    <            >
    \ J1 J2 J3 /

    :param j1:
    :param j2:
    :param j3:
    :param J1:
    :param J2:
    :param J3:
    :param print_warnings:
    :return:
    """

    # Check that the js and Js are only integer or half integer
    if ((2*j1 != round(2*j1)) |
        (2*j2 != round(2*j2)) |
        (2*j2 != round(2*j2)) |
        (2*J1 != round(2*J1)) |
        (2*J2 != round(2*J2)) |
        (2*J3 != round(2*J3))):
        print('All arguments must be integers or half-integers.')
        return -1
    
    # Check if the 4 triads ( (j1 j2 j3), (j1 J2 J3), (J1 j2 J3), (J1 J2 j3) )
    # satisfy the triangular inequalities
    if ((abs(j1-j2) > j3) | (j1+j2 < j3) |
        (abs(j1-J2) > J3) | (j1+J2 < J3) |
        (abs(J1-j2) > J3) | (J1+j2 < J3) |
        (abs(J1-J2) > j3) | (J1+J2 < j3)):
        if print_warnings:
            print('6j-Symbol is not triangular!')
        return 0
    
    # Check if the sum of the elements of each triad is an integer
    if ((2*(j1+j2+j3) != round(2*(j1+j2+j3))) |
        (2*(j1+J2+J3) != round(2*(j1+J2+J3))) |
        (2*(J1+j2+J3) != round(2*(J1+j2+J3))) |
        (2*(J1+J2+j3) != round(2*(J1+J2+j3)))):
        if print_warnings:
            print('6j-Symbol is not triangular!')
        return 0
    
    # Arguments for the factorials
    t1 = j1+j2+j3
    t2 = j1+J2+J3
    t3 = J1+j2+J3
    t4 = J1+J2+j3
    t5 = j1+j2+J1+J2
    t6 = j2+j3+J2+J3
    t7 = j1+j3+J1+J3

    # Finding summation borders
    tmin = max(0, max(t1, max(t2, max(t3, t4))))
    tmax = min(t5, min(t6, t7))
    tvec = np.arange(tmin, tmax + 1, 1)
        
    # Calculation the sum part of the 6j-Symbol
    WignerReturn = 0
    for t in tvec:
        WignerReturn += (-1) ** t * factorial(t + 1) / (factorial(t - t1) * factorial(t - t2) *
                                                        factorial(t - t3) * factorial(t - t4) *
                                                        factorial(t5 - t) * factorial(t6 - t) *
                                                        factorial(t7 - t))

    # Calculation of the 6j-Symbol
    w6j = WignerReturn * np.sqrt(triangle_coeff(j1, j2, j3) *
                                 triangle_coeff(j1, J2, J3) *
                                 triangle_coeff(J1, j2, J3) *
                                 triangle_coeff(J1, J2, j3))
    return w6j


def triangle_coeff(a, b, c):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    # Calculating the triangle coefficient
    return factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / (factorial(a + b + c + 1))
