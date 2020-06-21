"""
Calculate the dispersion of the 2S_{1/2} states of 6-lithium versus magnetic field and produce
a summary plot of the energies and eigenfunctions

atomic parameters taken from https://www.physics.ncsu.edu/jet/techdocs/pdf/PropertiesOfLi.pdf
"""
import datetime
from matrix_elem import *
import matplotlib.pyplot as plt

b_fields = np.linspace(0.01, 15, 200)
#b_fields = 33.8
saving = 0

# #######################################
# 6-lithium
atom_str = 'li-6'
state_str = '2S_{1/2}'
gs = 2.0023193043737
gl = 0.99999369
gi =-0.0004476540

atomic_params = [152.1368407, gs, gl, gi]
qnumbers = [0, 0.5, 1]

state_str_e =  '2P_{3/2}'
atomic_params_e = [-1.155, gs, gl, gi]
qnumbers_e = [1, 1.5, 1]

# #######################################

# diagonalize states for all magnetic fields
unc_basis = get_uncoupled_basis(qnumbers)
coup_basis = get_coupled_basis(qnumbers)
nstates = coup_basis.shape[0]

coupled_states_e = get_coupled_basis(qnumbers_e)
nstates_e = coupled_states_e.shape[0]

# solve and plot breit rabi problem
energies, unc_eigvects, coup_eigvects = breit_rabi(b_fields, atomic_params, qnumbers)

# branching ratios
_, branching_ratio_components, _, br_ratio_strs = get_branching_ratios(b_fields, atomic_params, qnumbers, atomic_params_e, qnumbers_e)
branching_ratios = branching_ratio_components[0] + branching_ratio_components[1] + branching_ratio_components[2]
#print br_ratio_strs[0]

# plot and save results
now = datetime.datetime.now()
now_str = "%04d;%02d;%02d_%02dh_%02dm" % (now.year, now.month, now.day, now.hour, now.minute)

energies_fig, eigstates_uncoupledfrac_fig, eigstates_coupledfrac_fig = \
    plot_states_vs_field(b_fields, energies, unc_basis, unc_eigvects, coup_basis, coup_eigvects, desc_str=atom_str, plot_log=1)

desc_str = "%s %s to %s" % (atom_str, state_str_e, state_str)
branching_ratios_fig = plot_branching_ratios(b_fields, branching_ratios, desc_str=desc_str, plot_log=1)

if saving:
    energies_fig.savefig('%s_energies_vs_field.png' % now_str)
    eigstates_uncoupledfrac_fig.savefig('%s_eigenstate_overlaps_with_uncoupled_basis_vs_field.png' %now_str)
    eigstates_coupledfrac_fig.savefig('eigenstate_overlaps_with_coupled_basis_vs_field.png')
    branching_ratios_fig.savefig('%s_branching_ratios_vs_field.png' % now_str)


plt.show()