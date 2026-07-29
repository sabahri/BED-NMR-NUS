# Bayesian Experimental Design for NMR Nonuniform Sampling

**DIANA** (Data-Informed Adaptive Nonuniform Acquisition): using Bayesian Experimental Design to decide, in real time, which evolution time points an NMR spectrometer should sample during a nonuniformly sampled (NUS) experiment.

## Project Overview

Nonuniform sampling (NUS) is a technique for multidimensional NMR that reduces the number of evolution time points collected, cutting experiment time while often improving sensitivity and resolution compared to uniform sampling. Conventionally, the sampling schedule is fixed in advance, generated from the size of the biological system, its isotopic labeling scheme, spin relaxation properties, and the information being targeted. That schedule cannot adapt once the experiment starts.

Machine learning already plays a large role in NMR — NUS reconstruction, structure calculation, shaped pulse optimization, chemical shift anisotropy modeling, chemical shift prediction — but always as a preparatory or post-processing step on data that has already been collected. To our knowledge, ML has not yet been used to let the spectrometer make independent, in-progress decisions about how to acquire data.

DIANA proposes to close that loop: a Bayesian model that learns, during acquisition, which time points are most informative, and adjusts the sampling trajectory on the fly using only the data collected so far — without prior knowledge of the biological system. The model reasons about the relationship between evolution time and the fundamental aspects of a spectrum (sensitivity, resolution, aliasing), and in principle can also learn where to stop, once additional points offer diminishing returns.

This is conceptually related to Targeted Acquisition (Orekhov and coworkers), which also collects NUS points "on the fly" and picks a stopping point from a spectral quality factor. The difference is that in Targeted Acquisition there is no feedback between the quality assessment and the choice of *which* points to sample next; DIANA is built specifically to close that loop. It also follows on from an earlier proof of concept using variational Bayesian analysis for active NUS (Worley, 2017), updated with more recent Bayesian estimation methods.

For the full derivation and motivation, see `DIANA_v9.docx`.

## Current Progress

This project currently simulates **1D acquisition**, intended to stand in for the indirect dimension of a 2D experiment. The pipeline runs end-to-end: it generates a synthetic FID, adaptively selects sampling points by entropy, and reconstructs a spectrum.

Evaluating reconstructed peak intensity as sampling progresses is not the right quality metric long-term, but it's currently used as a simple point of comparison against random sampling.

Simulation on a real 2D dataset (using Bruker `ser` files) is in progress — see `BED_NUS.py` for the file-handling groundwork (chunked binary reads, temporary `ser`/`acqu2s` manipulation via `nmrglue`) already laid for that step.

## Installation

This project depends on:

```
numpy
scipy
matplotlib
nmrglue
```

Install into a fresh environment with:

```bash
pip install numpy scipy matplotlib nmrglue
```

## Repository Contents

| File | Description |
|---|---|
| `bsnus25.ipynb` | Main working notebook: simulates a 1D FID, runs adaptive (entropy-based) NUS, compares it against exponentially-weighted random sampling. |
| `BED_NUS.py` | Refactored/consolidated functions from the notebook, plus Bruker `ser`/`acqu2s` file-handling utilities for running the procedure on real instrument data. |
| `DIANA_v9.docx` | Write-up of the motivation, math (Bayesian regression, entropy-based point selection), and methods. |

## Approach

### 1. Bayesian regression, mapped onto NMR

A single acquired time point is modeled as the inner product of a weight vector (the spectrum — relative intensities of the underlying frequency components) and a feature vector (a bank of candidate complex sinusoids), plus Gaussian noise. Treating spectrum recovery as Bayesian regression lets us reason about *the distribution of possible spectra* consistent with the points collected so far, rather than committing to a single point estimate.

We start with a **Gaussian prior over the weights** (i.e., ridge regression). This is a known simplification — real spectra are sparse, and a Gaussian prior isn't the most realistic choice — but it keeps the posterior available in closed form, which keeps the whole pipeline tractable while the core sampling-decision logic is worked out. A more realistic sparsity-promoting prior is a natural next step once the framework is validated.

### 2. Choosing new points by entropy

Sampling starts with a handful of points chosen at random (via inverse-CDF sampling of an exponential-decay-weighted distribution, biasing early points toward short evolution times). From there, at each step the algorithm computes the posterior predictive covariance for every unsampled time point and picks the one(s) that maximize entropy — i.e., where the model is currently most uncertain.

The key property of this criterion: it depends only on *which points have been sampled and how many total points exist*, not on the actual measured values. That means the next point to sample can be decided before that measurement is made — which is what makes this feasible as a live, in-the-loop acquisition decision rather than a purely retrospective analysis.

### 3. Modeling relaxation

Because NMR signal amplitude decays over the acquisition (T2 relaxation), an exponential decay envelope is fit to the sampled points as acquisition progresses, and used to correct for the fact that later time points are intrinsically noisier/weaker — so the entropy criterion isn't misled by relaxation alone into avoiding later points.

### 4. Simulate before generalizing

To keep the core sampling-decision logic tractable and debuggable, the method is first validated on simulated 1D data (standing in for a single indirect dimension) before extending to a real 2D dataset.

## References

1. Worley, B. Variational Bayesian analysis of nonuniformly sampled NMR data. *Concepts in Magnetic Resonance Part A* **2017**, *46A* (2), e21428. https://doi.org/10.1002/cmr.a.21428
2. O'Hagan, A. Bayes–Hermite quadrature. *Journal of Statistical Planning and Inference* **1991**, *29* (3), 245-260. https://doi.org/10.1016/0378-3758(91)90002-V
3. Jaravine, V. A.; Orekhov, V. Y. Targeted Acquisition for Real-Time NMR Spectroscopy. *J. Am. Chem. Soc.* **2006**, *128* (41), 13421-13426. DOI: 10.1021/ja062146p
