# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 23:38:58 2016

@author: Peter

Simulate number of photons scattering during absorption, including the effect of Doppler shifts as the atomic
velocity changes due to photon recoils
"""
from random import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt

save_results = 1

# set imaging time and intensity
tend = 10e-6
IoverIsat = 0.1

# helper function to get scattering rate
def ScattR(IIsat, Delta):
    return Gamma / 2 * IIsat / (1 + IIsat + 4 * pow(Delta / Gamma, 2))

# fundamental constants
hbar = 1.054e-34
mLi = 9.98834e-27

# 6-Li params
Isat = 25.4  # W_exp/m^2
Gamma = 2 * 3.14159 * 6e6
k = 2 * 3.14159 / 671e-9

# simulation parameters
dt = 1e-9
NRepeats = 1000  # number of times to repeat the simulation

# estimate total number of photons that would be scattered ignoring doppler shifts
NPhotonsEstimate = ScattR(IoverIsat, 0) * tend
# estimate the most likely imaging freuqency. Since we optimize this experimentally
# most likely we actually park the laser at half of the net doppler shift
DeltaStart = NPhotonsEstimate / 2.0 * (hbar * k / mLi) * k

# loop for doing many simulations
ScatteredPhotonList = []
for ii in range(NRepeats):
    # single simulation
    CurrentDelta = -DeltaStart
    TotalScatteredPhotons = 0.0

    for t in np.arange(0, tend, dt):
        # Poisson distribution estimate of how many multiple photon events are possible
        GammaEff = ScattR(IoverIsat, CurrentDelta)
        Prob0 = exp(-GammaEff * dt)
        Prob1 = Gamma * dt * exp(-GammaEff * dt)
        Prob2 = pow(GammaEff * dt, 2) / 2 * exp(-GammaEff * dt)

        BoolScatteredPhoton = (random() > Prob0)  # randomly choose if have scatter a photon
        if BoolScatteredPhoton:  # if a photon is sccattered, count it and add doppler shift to the detuning
            TotalScatteredPhotons = TotalScatteredPhotons + 1
            CurrentDelta = CurrentDelta + (hbar * k / mLi) * k  # doppler shift
        else:
            pass

    ScatteredPhotonList.append(TotalScatteredPhotons)

AvgPhotons = sum(ScatteredPhotonList) / len(ScatteredPhotonList)
ScatteredPhotonReduction = AvgPhotons / NPhotonsEstimate
print("Average number of photons scattered = %0.2f" % AvgPhotons)
print("Fractional reduction in number of photons scattered versus infinitely massive atom (no doppler shifts) = %0.3f" % ScatteredPhotonReduction)

fig_handle = plt.figure()
plt.hist(ScatteredPhotonList)
plt.xlabel('Scattered Photons')
plt.ylabel('Occurrences')
plt.title('Avg Photons = %0.1f, Scattered Photon Reduction = %0.2f' % (AvgPhotons, ScatteredPhotonReduction))

if save_results:
    fig_handle.savefig('scattered_photons_histogram.png')

plt.show()