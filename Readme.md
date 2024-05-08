
# PySurvMC

PySurvMC is a Python package designed for Bayesian survival analysis using Markov Chain Monte Carlo (MCMC) methods. This package provides a user-friendly interface to build survival models with left, right, and interval censoring, and allows for the integration of arbitrary prior distributions.

## Features

- **Support for Multiple Censoring Types**: Handle left, right, and interval-censored data efficiently.
- **Flexible Prior Specification**: Users can specify their own prior distributions for model parameters.
- **Built-in Models**: Includes common survival models like the proportional hazard and accelerated failure time models.
- **Advanced MCMC Sampling**: Utilizes the No-U-Turn Sampler (NUTS), a variant of the Hamiltonian Monte Carlo (HMC), for efficient parameter estimation.

## Installation

PySurvMC is available on PyPI and can be installed using pip:

```bash
pip install PySurvMC
```

## Usage

Below is a simple example of how to use PySurvMC to fit a Weibull proportional hazard model:

```python
from PySurvMC import WeibullPH
import pymc as pm

# Define priors
priors = {
    "coeff": pm.Normal.dist(mu=0, sigma=100, shape=len(covariates)),
    "shape": pm.HalfNormal.dist(sigma=100),
    "scale": pm.HalfNormal.dist(sigma=100)
}

# Initialize model
weiph = WeibullPH(lb=lb, ub=ub, event=event, covariates=covariates, priors=priors)

# Fit model
weiph_mcmc.fit(draws=10000, cores=8)
```

## Documentation

For more detailed documentation, please visit our [GitHub repository](https://github.com/yueht23/PySurvMC).

## Examples

The package includes examples and a detailed simulation study that evaluates its performance against frequentist approaches, as well as a real-world case study using the North Central Cancer Treatment Group Lung cancer dataset.

## Contributing

Contributions to PySurvMC are welcome! Please read our contribution guidelines on our [GitHub repository](https://github.com/yueht23/PySurvMC) to get started.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors

- **[Yueht23]** - *Initial work* - [GitHub](https://github.com/yueht23)

## Acknowledgments

- Thanks to the contributors of the PyMC libraries which are extensively used in this project.
---

