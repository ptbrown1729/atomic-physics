import numpy as np
import matplotlib.pyplot as plt
from matrix_elem import *

c = 299792458  # m/s
# atomic parameters for 6-Li D1 And D2 lines
lambda_d1 = 670.992421e-9
lambda_d2 = 670.977338e-9
lambda_d = 3 * c / ( 2 * c / lambda_d2 + c / lambda_d1)
gamma_d1 = (2 * np.pi) * 5.8724e6
gamma_d2 = (2 * np.pi) * 5.8724e6
gs = 2.0023193043737
gl = 0.99999369
gi =-0.0004476540

# get reduced matrix elements from decay rates
d2_redj, d_redl = gamma2matrixel_Dline(lambda_d2, 1.5, gamma_d2, convention="wigner")
d1_redj, d_redl_check = gamma2matrixel_Dline(lambda_d1, 0.5, gamma_d1, convention="wigner")

# laser parameters
b_fields = 0.0001 #np.linspace(0.01, 600, 100)
laser_detune = -5e9
lambda_laser = c / (c / lambda_d1 + laser_detune)
intensity =  1e-6 / (750e-9**2 * 100)

#reduced_matrixel_l = 1.148e-29 # C * m, in Wigner convention
# [Ahf, gs, gl, gi]
params_g = [152.1368407, gs, gl, gi]
params_d1 = [17.386, gs, gl, gi]
params_d2 = [-1.155, gs, gl, gi]

# solve for state-dependent light-shifts
I = 1
lshifts, pols, gammas = get_lightshift_Dline(b_fields, lambda_d1, lambda_d2, lambda_laser, intensity,
                         I, params_g, params_d1, params_d2, d_redl, convention='wigner')

# for testing purposes, solve Breit-Rabi problem
# ground state, eigenstates
qnumbers_g = [0, 0.5, I]
qnumbers_d1 = [1, 0.5, I]
qnumbers_d2 = [1, 1.5, I]

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

# semiclassical approximation neglecting spin dependence
ls_semi, gamma_semi = get_lightshift_semiclass( lambda_d, 0.5 * (gamma_d1 + gamma_d2), lambda_laser, intensity)
ls_semi = ls_semi / 6.626e-34 / 1e6

# [gs, gl], [ss, ll, jj]
gj = get_gj([gs, gl], [0.5, 0, 0.5])
# [gj, gi], [jj, ii, ff]
gf_half = get_gf([gj, gi], [0.5, 1, 0.5])
ls_semi_fhalf, _, basis_half = get_all_lightshift_semiclass(lambda_d1, lambda_d2, gamma_d1, lambda_laser, intensity, gf_half, 0.5)

gf_thalf = get_gf([gj, gi], [0.5, 1, 1.5])
ls_semi_fthreehalf, _, basis_threehalf = get_all_lightshift_semiclass(lambda_d1, lambda_d2, gamma_d1, lambda_laser, intensity, gf_thalf, 1.5)

# reorder to match breit-rabi basis ordering
ls_semi_pol = np.concatenate(( np.flip(ls_semi_fhalf, axis=0), ls_semi_fthreehalf))[:, None] / 6.626e-34 / 1e6
basis = np.concatenate(( np.flip(basis_half, axis=0), basis_threehalf), axis=0)


# TODO: need to check these values. Some issues...need to test that the semiclassical result agrees with the other calc
# in the limit where the detuning is large compared with the hyperfine splitting ... right now they are giving me
# opposite shifts for the F=3/2 ground states between the two techniques...
print lshifts[:, 0, :]
print ls_semi
print np.squeeze(ls_semi_pol)

# plot results
diff_12_sigp = lshifts[0, :, 1] - lshifts[1, :, 1]
diff_23_sigp = lshifts[1, :, 1] - lshifts[2, :, 1]
diff_13_sigp = lshifts[0, :, 1] - lshifts[2, :, 1]

plt.figure()
plt.plot(b_fields, np.abs(diff_13_sigp) * 1e3)
plt.plot(b_fields, np.abs(diff_12_sigp) * 1e3)
plt.plot(b_fields, np.abs(diff_23_sigp) * 1e3)
plt.xlabel('Magnetic Field (G)')
plt.ylabel('Differential lightshift (KHz)')
plt.title('Differential lightshift magnitude between |1>, |2>, and |3> Vs. Bfield\n Detuning=%0.2f GHz from D1 line, intensity = %0.2f W/m**2' % (laser_detune/1e9, intensity))
plt.legend(['13', '12', '23'])