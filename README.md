# Approximation and Optimization Theory for Linear Continuous Time RNNs

Code and numerical experiments accompanying my bachelor's thesis.

[Read the thesis](thesis.pdf)

The main theoretical reference is Li et al. [1], whose approximation and optimization results provide the basis for the two continuous time experiments.

The repository contains three experiments: a discrete time copy-first memory task, kernel approximation for two smoothness classes, and gradient flow plateaus for increasing memory length. The experiments are numerical illustrations over finite parameter ranges. They do not prove the approximation or optimization results discussed in the thesis.

## Experiments

| Experiment                                                      | Thesis location | Implementation and notebook |
|-----------------------------------------------------------------| --- | --- |
| Copy-first memory                                               | Chapter 2, Figure 1 | [Python script](experiments/01_copy_first_memory.py) |
| Approximation rates for analytic and limited-smoothness kernels | Section 5.3.3, Proposition 5.10, Figures 2–4 | [Python implementation](experiments/04_theorem10_approximation_rates.py), [notebook](notebooks/03_theorem10_approximation_rates.ipynb) |
| Gradient flow plateaus for delayed kernels                      | Section 6.5, Table 2, Figures 5–7; numerical validation in Appendix 8.2 | [Python implementation](experiments/08_theorem20_exact_gradient_flow.py), [notebook](notebooks/08_theorem20_plateauing.ipynb) |

The notebook filenames follow the theorem numbering in Li et al. [1]: Theorem 10 corresponds to Theorem 5.9 in the thesis, and Theorem 20 to Theorem 6.5. The target choices and numerical results are my own implementations of the settings discussed in the thesis.

## Setup

The existing project environment uses Python 3.11. The pinned packages in `requirements.txt` record that environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The main numerical dependencies are NumPy, SciPy, Matplotlib, and PyTorch. The notebooks also use Jupyter and, for the plateau experiment, pandas. Installing the full requirements file also installs the recorded notebook environment.

## Running the experiments

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open either notebook listed above and run its cells in order. Starting Jupyter inside `notebooks/` is also supported.

### Copy-first memory

```bash
python experiments/01_copy_first_memory.py
```

The task follows the long term dependency setting of Bengio et al. [2]. A one-layer tanh RNN with hidden dimension 16 is implemented in PyTorch [5] and trained using Adam [6] for 20 epochs at sequence lengths 4, 10, 20, and 50. The script sets the PyTorch seed to 0 and uses one training run per length. It saves the test-loss figure as PNG and PDF under `outputs/figures/`. These individual runs are illustrative; they do not provide uncertainty estimates across random seeds.

### Approximation rates

Run `notebooks/03_theorem10_approximation_rates.ipynb` to reproduce the thesis presentation. It imports the implementation, recomputes the least-squares fits and quadrature checks, and saves the three thesis PNG figures under `figures/`.

Both targets use the same theoretical parameters, alpha = beta = 1, and widths 2 through 16. The exponential rates are fixed by the proof construction for Theorem 10 in Li et al. [1], while coefficients are fitted by discrete least squares. Errors are evaluated in L1 on the full interval [0, infinity). The notebook reports normalized errors and numerical consistency checks. Its empirical constants are not estimates of the unknown universal constant in the approximation theorem.

The standalone implementation can also be run with:

```bash
python experiments/04_theorem10_approximation_rates.py
```

It additionally produces diagnostic plots and a results table under `outputs/`.

### Gradient-flow plateaus

Run `notebooks/08_theorem20_plateauing.ipynb` to reproduce Table 2, the three thesis figures, and the validation summary. The model width is 10 and the delays are 8, 10, 12, 14, 16, and 18. The loss and gradients are evaluated in closed form on [0, infinity), and the gradient flow is integrated using SciPy's DOP853 solver with event detection [3, 4].

The committed cache is `notebooks/.cache/08_theorem20_plateauing_results.pkl`. The notebook checks its schema version and the SHA-256 digest of the experiment source. With a matching cache, it reports:

```text
Cache status: loaded from cache; no solver was called
```

It then reuses the six primary, six refined, and three illustrative trajectories and their stored validation values. If the cache is missing or the experiment source changes, the notebook recomputes these results and replaces the cache. Recalculation can take substantially longer than plotting the stored results. Loading cached validation values does not independently repeat the validation.

The notebook saves PNG and vector PDF versions of the target-kernel, plateau-trajectory, and learned-kernel figures under `figures/`.

The standalone implementation can also be run with:

```bash
python experiments/08_theorem20_exact_gradient_flow.py
```

This performs a fresh computation, including its self-checks, primary/refined solver comparison, and additional diagnostic plots. It does not use the notebook cache.

## Numerical validation

The closed-form plateau implementation checks its initialization, compares its gradient with centered finite differences, compares its loss with independent quadrature, and compares hitting times from primary and refined solver settings. The derivation and checks are documented in Appendix 8.2 of the thesis. These checks assess numerical consistency and sensitivity to solver settings; they do not verify an asymptotic theorem.

The earlier implementation, `experiments/07_theorem20_exponential_plateau.py`, is retained because `tests/test_theorem20_numerics.py` imports it directly. The tests examine quadrature-grid and horizon sensitivity, Euler step-size refinement, final rate bounds, and trapezoidal versus Simpson quadrature. They exercise implementation 07; the notebook uses implementation 08 and its own validation.

To run these additional tests, install pytest in the project environment:

```bash
python -m pip install pytest
python -m pytest tests/test_theorem20_numerics.py
```

Some tests run shortened Euler simulations, so they take longer than loading the plateau notebook's cache.

## References

**[1]** Zhong Li, Jiequn Han, Weinan E, and Qianxiao Li (2022). *Approximation and Optimization Theory for Linear Continuous-Time Recurrent Neural Networks*. Journal of Machine Learning Research, 23(42), 1–85. [JMLR record](https://jmlr.org/papers/v23/21-0368.html). Theorems 10 and 20 provide the theoretical settings for the two continuous-time experiments.

**[2]** Yoshua Bengio, Patrice Simard, and Paolo Frasconi (1994). *Learning Long-Term Dependencies with Gradient Descent Is Difficult*. IEEE Transactions on Neural Networks, 5(2), 157–166. [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181).

**[3]** Pauli Virtanen et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17, 261–272. [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

**[4]** Ernst Hairer, Syvert P. Nørsett, and Gerhard Wanner (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Section II.5, pp. 181–185. [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1). Reference for DOP853.

**[5]** Adam Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems, 32. [Paper](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

**[6]** Diederik P. Kingma and Jimmy Ba (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015. [Updated arXiv version cited in the thesis](https://arxiv.org/abs/1412.6980v9).
