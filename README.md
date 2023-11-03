# JaxPeriodDrwFit


This project aims to find and characterize potential periodic signals in AGN (active galactic nuclei) light curves. We initially aim to re-analyze available datasets, such as ZTF (Zwicky Transient factory) and Pan-STARRS. The added value we are looking to add is the speed of analysis, done primarily through two avenues:

* use of parallel computing (TAPE project)
* acceleration of code via JAX

The files contained are:
* Create.ipynb - Notebook to create simulated ZTF data with known properties
* Fit.ipynb - Notebook to fit the data with stochastic + periodic component
* Analyze.ipynb - Notebook for analysis of the results from the fitting algorithms
* Visualize.ipynb - Notebook to show the results of the analysis
* JaxPeriodDrwFit.py - Main dataclass 
