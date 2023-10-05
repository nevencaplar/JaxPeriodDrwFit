# JaxPeriodDrwFit


This project aims to find and characterize potential periodic signals in AGN (active galactic nuclei) light curves. We initially aim to re-analyze available datasets, such as ZTF (Zwicky Transient factory) and Pan-STARRS. The added value we are looking to add is the speed of analysis, done primarily through two avenues:

* use of parallel computing (TAPE project)
* acceleration of code via JAX

The files contained are:
* Analysis.ipynb - Analysis of the results from the fitting algorithms
* Drw_per.py - Script to analyze the data, with TAPE
* Drw_per_i.ipynb - Interactive notebook for exploration
* Drw_per_no_tape.py - Script to analyze the data, without TAPE
* JaxPeriodDrwFit.py - Main dataclass
* create_data - Script to generate fake data with known variability parameters 
