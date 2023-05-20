"""
Calculate RF transitions vs. magnetic field for 6-lithium in the 2S_{1/2} state.
see https://www.physics.ncsu.edu/jet/techdocs/pdf/PropertiesOfLi.pdf for information about atomic units
"""

import numpy as np
import matplotlib.pyplot as plt
import atomic_physics.matrix_elem as mel

# 6-lithium
# 2S_{1/2}
atomic_params = [152.1368407, 2.0023193043737, 0.99999369, -0.0004476540]
qnumbers = [0, 0.5, 1]

#b_fields = np.arange(0.1, 1000, 10)
b_fields = np.arange(1.6, 2, 0.02)

nstates = int((2 * qnumbers[1] + 1) * (2 * qnumbers[2] + 1))
energies_all_fields = np.zeros((len(b_fields), nstates))
eigvects_coupled_all_fields = np.zeros((nstates, nstates, len(b_fields)))
eigvects_uncoupled_all_fields = np.zeros((nstates, nstates, len(b_fields)))
# breit rabi solution for each field
for ii, b_field in enumerate(b_fields):
    energies, eigvects_uncoupled, eigvects_coupled = mel.breit_rabi(b_field, atomic_params, qnumbers)
    energies_all_fields[ii, :] = energies[:, 0]
    eigvects_uncoupled_all_fields[:, :, ii] = eigvects_uncoupled[:, :, 0]
    eigvects_coupled_all_fields[:, :, ii] = eigvects_coupled[:, :, 0]

# now get transitions
transitions = np.zeros((nstates, nstates, len(b_fields)))
for ii in range(0, nstates):
    for jj in range(0, nstates):
        transitions[ii, jj, :] = energies_all_fields[:, ii] - energies_all_fields[:, jj]

figh = plt.figure()
plt.plot(b_fields, transitions[0, 1, :])
plt.plot(b_fields, transitions[2, 3, :])
# plt.plot(b_fields, transitions[2, 4, :])
# plt.plot(b_fields, transitions[2, 5, :])
plt.grid()
plt.xlabel('Field (G)')
plt.ylabel('Transition Frequency (MHz)')
# plt.legend(['12', '34', '35', '36'])
plt.legend(['12', '34'])
plt.show()
