# atomic physics tools
Various tools for doing atomic physics calculations, including calculating eigenstates of the 
hyperfine and Zeeman  interactions versus magnetic field (i.e. solving the Breit-Rabi problem),
calculating branching ratios for alkali atoms decaying from the D1 or D2 transition, and tools
for solving the  optical Bloch equations.

Excellent references on the properties of various alkali atoms are available online.
Particularly useful are Michael Steck's sodium, rubidium, and cesium [datasheets](https://steck.us/alkalidata/),
Tobias Tiecke's [potassium datasheet](https://www.tobiastiecke.nl/archive/PotassiumProperties.pdf),
and Michael Gehm's [lithium-6 datasheet](https://www.physics.ncsu.edu/jet/techdocs/pdf/PropertiesOfLi.pdf).

## [matrix_elem.py](atomic_physics/matrix_elem.py)
Calculate the angular portion of atomic matrix elements. Tests which compare
the results of these functions and published results are found in
[matrix_elem_unittest.py](matrix_elem_unittest.py).

## [wigner.py](atomic_physics/wigner.py)
Tools for calculating the Clebsch-Gordon coefficients,
[Wigner 3j](https://mathworld.wolfram.com/Wigner3j-Symbol.html),
and [Wigner 6j](https://mathworld.wolfram.com/Wigner6j-Symbol.html) symbols.
The functions here have been taken from other sources and adapted to python. The 
[Clebsch-Gordon coefficients](https://www.mathworks.com/matlabcentral/fileexchange/5276-clebschgordan-m)
and [Wigner 3j symbol](https://www.mathworks.com/matlabcentral/fileexchange/5275-wigner3j-m?s_tid=prof_contriblnk)
functions were originally written by David Terr and are available on the MathWorks file exchange. 
The [Wigner 6j symbol]() function was written by Ulrich Krohn.

## [numerov.py](atomic_physics/numerov.py)
Tools for solving differential equations using the Numerov method. This is relevant to atomic physics
for solving the radial wavefunction of Rydberg atoms. The approach here was adopted
from 

## [examples](examples)
Example scripts utilizing, mostly for calculating properties of lithium-6.

## [tests](tests)
Tests verifying matrix element calculations

## [optical_bloch_eqns](optical_bloch_eqns)
Several Matlab functions for solving the optical Bloch equations and
calculating properties of electromagnetically induced transparency (EIT).
