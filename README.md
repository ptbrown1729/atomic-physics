# atomic physics tools
Various tools for doing atomic physics calculations, including calculating eigenstates of the 
hyperfine and Zeeman interactions versus magnetic field (i.e. solving the Breit-Rabi problem),
calculating branching ratios for alkali atoms decaying from the D1 or D2 transition, and tools
for solving the  optical Bloch equations.

Excellent references on the properties of various alkali atoms are available online.
- Daniel Steck's sodium, rubidium, and cesium [datasheets](https://steck.us/alkalidata/),
- Tobias Tiecke's [potassium datasheet](https://www.tobiastiecke.nl/archive/PotassiumProperties.pdf),
- Michael Gehm's [lithium-6 datasheet](https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf).

Useful background reading
- Daniel Steck's atomic physics [notes](https://atomoptics.uoregon.edu/~dsteck/teaching/quantum-optics/).
- J.T.M. Walraven's [notes](https://staff.fnwi.uva.nl/j.t.m.walraven/walraven/Lectures.htm)
- [Optical molasses and multilevel atoms: theory](https://doi.org/10.1364/JOSAB.6.002058)

## [matrix_elem.py](atomic_physics/matrix_elem.py)
Calculate the angular portion of atomic matrix elements.

## [wigner.py](atomic_physics/wigner.py)
Tools for calculating the Clebsch-Gordon coefficients,
[Wigner 3j](https://mathworld.wolfram.com/Wigner3j-Symbol.html),
and [Wigner 6j](https://mathworld.wolfram.com/Wigner6j-Symbol.html) symbols.
The functions here have been taken from other sources and adapted to python. The 
[Clebsch-Gordon coefficients](https://www.mathworks.com/matlabcentral/fileexchange/5276-clebschgordan-m)
and [Wigner 3j symbol](https://www.mathworks.com/matlabcentral/fileexchange/5275-wigner3j-m?s_tid=prof_contriblnk)
are based on functions created by David Terr which are available on the MathWorks file exchange. 
The Wigner 6j symbol function was based on a function written by Ulrich Krohn.

## [numerov.py](atomic_physics/numerov.py)
Tools for solving differential equations using the Numerov method. This is relevant to atomic physics
for solving the radial wavefunction of Rydberg atoms. The approach here was adopted
from 

## [examples](examples)
Example scripts utilizing, mostly for calculating properties of lithium-6.

## [tests](tests)
Tests verifying matrix element calculations by comparing with published results

## [optical_bloch_eqns](optical_bloch_eqns)
Several Matlab functions for solving the optical Bloch equations and
calculating properties of electromagnetically induced transparency (EIT).
