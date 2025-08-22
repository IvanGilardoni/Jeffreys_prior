# Jeffreys_prior

This project demonstrates how to perform **Bayesian sampling of ensembles**, emphasizing the critical role of the **Jeffreys prior** in Bayesian inference. The notebooks and scripts provide a hands-on exploration of Bayesian techniques, focusing on how ensemble counting (namely, choosing the non-informative reference prior) influences results.

---

## Project Structure

```
Jeffreys_prior/
├── Functions/
│ ├── basic_functions_bayesian.py # Core Bayesian functions used throughout the project (old version, the new one has been included in `MDRefine`).
│ ├── basic_functions_gaussian.py # Supporting functions for the simple Gaussian model.
│ ├── coretools.py # Core functions including the definition of the class Result.
│
├── Manuscript_images/ # images for the manuscript.
│
├── main_notebook_1.ipynb/ # Introduction and initial Bayesian sampling examples demonstrating core concepts on a Gaussian toy-model. 
├── main_notebook_2.ipynb/ # Analysis of sampling outputs for the realistic case of RNA refinement (including results from calculations performed on the Ulysses cluster).
├── main_sampling_bayesian.py/ # Script that utilizes the above functions to perform Bayesian sampling on the Ulysses cluster. Its outputs are further analyzed in `main_notebook_2.ipynb`.
└── README.md # This file
```

---

## Getting Started

### Requirements

- Python 3.8+
- Jupyter Notebook
- Required packages (install via pip):  

```
pip install numpy scipy matplotlib jupyter
```

### Basic functions

1. `basic_functions_bayesian.py`: old version, the new one is [`MDRefine/bayesian.py`](https://github.com/bussilab/MDRefine/blob/master/MDRefine/bayesian.py).

2. `basic_functions_gaussian.py`: supporting functions for the simple Gaussian model.

### Running the notebooks

1. Clone the repository:

```
git clone https://github.com/yourusername/Jeffreys_prior.git
cd Jeffreys_prior
```

2. Launch Jupyter Notebook:

```
jupyter notebook
```

3. Open and run [`main_notebook_1.ipynb`](main_notebook_1.ipynb) to get introduced to Bayesian sampling with different non-informative priors on a simple Gaussian toy-model.

4. Use [`main_sampling_bayesian.py`](main_sampling_bayesian.py) to perform the ensemble sampling on your local machine.

5. Analyze the outputs in [`main_notebook_2.ipynb`](main_notebook_2.ipynb), which demonstrates the impact of the Jeffreys non-informative prior on the results.

---

## About the Jeffreys prior

The Jeffreys prior is a non-informative prior derived from the Fisher information metric, providing an objective Bayesian prior that is invariant under reparameterization. This project showcases how using the Jeffreys prior can lead to more robust inference in ensemble sampling, particularly when compared to other non-informative priors.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer at `igilardo@sissa.it`.