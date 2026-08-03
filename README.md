# Bayesian Experimental Design for NMR Nonuniform Sampling

Using Bayesian Experimental Design to decide, in real time, which evolution time points an NMR spectrometer should sample during a nonuniformly sampled (NUS) experiment.

## Project Overview

Nonuniform sampling (NUS) is a technique for multidimensional NMR that reduces the number of evolution time points collected, cutting experiment time while often improving sensitivity and resolution compared to uniform sampling. Conventionally, the sampling schedule is fixed in advance, and cannot adapt once the experiment starts. To our knowledge, ML has not yet been used to let the spectrometer make independent, in-progress decisions about how to acquire data.

Here we work on a Bayesian model that learns, during acquisition, which time points are most informative. It then adjusts the sampling trajectory using only the data collected so far.

## Project Status

This project currently simulates **1D acquisition**, intended to stand in for the indirect dimension of a 2D experiment. The pipeline generates a synthetic FID, adaptively selects sampling points by entropy, and reconstructs a spectrum.

Evaluating reconstructed peak intensity as sampling progresses is not the right quality metric long-term, but it's currently used as a simple point of comparison against random sampling.

Simulation on a real 2D dataset (using Bruker `ser` files) is in progress. See `BED_NUS.py` for the file-handling groundwork (chunked binary reads, temporary `ser`/`acqu2s` manipulation via `nmrglue`) already laid for that step.

**Related work:** since this project was last active, Worley has published an active nonuniform sampling (ANS) algorithm (VRLS) that is fast enough to run within an acquisition's recycle delay, addressing the practicality gap that blocked his earlier variational approach (Worley, 2026; see References). VRLS uses a Laplace (sparsity-promoting) prior on the spectrum rather than the Gaussian prior used here, which is also the direction planned for this project.

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

## Approach

### 1. Bayesian regression, mapped onto NMR

A single acquired time point is modeled as the inner product of a weight vector (the relative intensities of the underlying frequency components) and a feature vector (a bank of candidate complex sinusoids), plus Gaussian noise:

$$y = \Phi w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}_{\mathbb{C}}(0, \sigma^2 I)$$

where $\Phi$ is the $N \times D$ design matrix built from the $N$ sampled time points and $D$ candidate frequencies, and $w \in \mathbb{C}^D$ is the (unknown) spectrum. Treating spectrum recovery as Bayesian regression lets us reason about *the distribution of possible spectra* consistent with the points collected so far, rather than committing to a single point estimate.

We start with a **Gaussian prior over the weights**, $w \sim \mathcal{N}_{\mathbb{C}}(0, \chi^2 I)$ (i.e., ridge regression), which is conjugate and gives a closed-form Gaussian posterior:

$$p(w \mid y) = \mathcal{N}_{\mathbb{C}}(m, S), \qquad S = \left(\Phi^{\mathsf{H}}\Phi + \lambda I\right)^{-1}, \qquad m = S\,\Phi^{\mathsf{H}} y, \qquad \lambda = \sigma^2/\chi^2$$

This is a known simplification: real spectra are sparse, and a Gaussian prior isn't the most realistic choice. However it keeps the posterior tractable in closed form. A more realistic sparsity-promoting prior is a natural next step once the framework is validated (see Worley reference below).

**A side effect of this choice of prior:** the posterior covariance $S$ depends only on $\Phi$ (i.e., on *which* time points were sampled) and on $\lambda$, never on the measured values $y$. Only the posterior mean $m$ depends on $y$. This is a consequence of the model being linear-Gaussian, not a design goal.

### 2. Choosing new points by entropy

Sampling starts with a handful of points chosen at random (via inverse-CDF sampling of an exponential-decay-weighted distribution, biasing early points toward short evolution times). From there, at each step the algorithm scores every unsampled candidate point $t^\star$ by how much it would reduce uncertainty about the spectrum.

The predictive distribution for a candidate $t^\star$ follows directly from the weight-space posterior $S$ in section 1: $S$ is built from the *sampled* points only, not the candidate. Since $y^\star = \phi(t^\star)^{\mathsf{T}}w + \varepsilon^\star$ is linear in $w$, its predictive variance is a single quadratic form:

$$\tilde{\Sigma}(t^\star) = \phi(t^\star)^{\mathsf{T}} S\, \overline{\phi(t^\star)} + \sigma^2$$

and its entropy (complex scalar Gaussian) is:

$$H(t^\star) = \log(\pi e) + \log \tilde{\Sigma}(t^\star)$$

The next point sampled is $t_{\text{next}} = \arg\max_{t^\star} H(t^\star)$ over the unsampled candidates. Equivalently, since $\log$ is increasing, the candidate with the largest predictive variance.

This can equivalently be derived by appending the candidate's feature row to the design matrix,

$$\Phi^\star = \begin{bmatrix}\Phi \\ \phi(t^\star)^{\mathsf{T}}\end{bmatrix},$$

forming the joint predictive covariance $\Sigma_{\text{pred}} = \Phi^\star S \Phi^{\star\mathsf{H}} + \sigma^2 I$ over sampled points and candidate together, and taking the Schur complement after partitioning into the sampled block, the candidate's own variance, and their cross-covariance. The two routes agree exactly, but the direct quadratic form above is cheaper and avoids an unnecessary $(N{+}1)\times(N{+}1)$ matrix.

A consequence of this criterion: $\tilde{\Sigma}(t^\star)$ (and therefore $H$) is a function of $\Phi$ and $\phi(t^\star)$ alone, so it depends only on *which points have been sampled*, not on the actual measured values. In code this is `XSXT()`/`Q()`.

This is a limitation of the current Gaussian-prior model, not the intent. The goal is for the history of measured intensities to actively shape which point gets sampled next e.g., a sparsity-promoting prior would make the posterior covariance depend on $y$ through the current estimate of which frequencies are actually present, so that measurements already taken determine how future ones are collected (see section 1's note on moving past the Gaussian prior).

Under the equal-variance Gaussian prior above, on the full integer frequency grid, the feature vectors $\phi(t)$ for different sampled times are mutually orthogonal. That makes different measurements statistically independent, which collapses $\tilde{\Sigma}(t^\star)$ to the same constant value for every unsampled candidate. As a result, the criterion has nothing to discriminate on. In the current code, the `alpha` frequency-grid scaling factor is what breaks this orthogonality and gives the entropy landscape any structure at all. A better way is to break the symmetry deliberately: either per-component decay rates in $\phi(t)_d = e^{(i\omega_d - \rho_d)t}$, or the unequal-variance/sparsity-promoting prior already planned above.

### 3. Modeling relaxation

Because NMR signal amplitude decays over the acquisition (T2 relaxation), an exponential decay envelope $A(t) = A_0 e^{-t/T_2}$ can be fit to the sampled magnitudes and folded into the predictive covariance, so that later, weaker time points aren't misjudged as more "informative" than they really are. This relaxation-weighted entropy is implemented as `T2kern_entropy()` in the notebook, but it is **not** the version currently used in the main sampling loop, which calls the plain, y-independent `Q()` from section 2 above. So relaxation is not yet actually affecting point selection in the results produced so far.

### 4. Simulate before generalizing

To keep the core sampling-decision logic tractable and debuggable, the method is first validated on simulated 1D data (standing in for a single indirect dimension) before extending to a real 2D dataset.

## References

1. Worley, B. Variational Bayesian analysis of nonuniformly sampled NMR data. *Concepts in Magnetic Resonance Part A* **2017**, *46A* (2), e21428. https://doi.org/10.1002/cmr.a.21428
2. O'Hagan, A. Bayes–Hermite quadrature. *Journal of Statistical Planning and Inference* **1991**, *29* (3), 245-260. https://doi.org/10.1016/0378-3758(91)90002-V
3. Jaravine, V. A.; Orekhov, V. Y. Targeted Acquisition for Real-Time NMR Spectroscopy. *J. Am. Chem. Soc.* **2006**, *128* (41), 13421-13426. DOI: 10.1021/ja062146p
4. Worley, B. Active nonuniform sampling. *Journal of Magnetic Resonance* **2026**, *386*, 108046. https://doi.org/10.1016/j.jmr.2026.108046
