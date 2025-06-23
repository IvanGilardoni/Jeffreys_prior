# Jeffreys_prior

The notebooks `main_notebook_1.ipynb` and `main_notebook_2.ipynb` are the central filess of this github project `Jeffreys_prior`, along with the Python functions `Functions/basic_functions_bayesian.py` and `Functions/basic_functions_gaussian.py` and the Python file `main_sampling_bayesian.py` that employs these two functions for calculations on Ulysses cluster whose output are analyzed in a section of `main_notebook_2.ipynb`.

The notebook `MD_analysis.ipynb` should not stay in this github project, since it refers to a different work.
It analyses some output data of MD simulations of RNA oligomers, in particular the relative distances between (C5 atoms of) each nucleotide, clustering in ensemble conformations, and the H-bonds between nucleotides. The cluster analysis can be extended to a Markov State Model, deriving transition paths and rates between clusters. Moreover, the H-bonds can be analyzed also in relation with clusters. Then, you can extract images illustrating these different conformations (including the formation of Hydrogen bonds, either Watson-Crick or non-canonical).
