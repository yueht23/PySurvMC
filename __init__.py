import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import warnings


class ProportionalHazard():
    """
    PoportionalHazard class a abstract class which is used to fit the survival data to the different distributions,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the ProportionalHazard class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        self.lb = lb
        self.ub = ub
        self.event = event
        self.covariates = covariates

        # check the priors, whether the all priors are given or not
        assert "coeff" in priors, "The prior of the coefficients is not given"

        self.coefficients = priors["coeff"]

        # initialize the scale parameter by propotionality assumption
        self.scale = pm.math.exp(-pm.math.dot(self.covariates, self.coefficients))

        # need to initialize by the child class
        self.shape = None
        self.rv = None

    def _logp(self, lb, ub, event, rv):
        """
        The log likelihood function of the Generalized Proportional Hazard model.
        :parameter
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        rv : PyMC3 distribution
            The random variable of the distribution
        """
        llk = (event == -1) * pm.logcdf(rv, ub)
        llk += (event == 0) * pm.logp(rv, lb)
        llk += (event == 1) * (1 - pm.logcdf(rv, lb))
        llk += (event == 2) * (pm.logcdf(rv, ub) - pm.logcdf(rv, lb))
        return llk

    def fit(self, draw=1000, tune=100, chains=4, cores=4, nuts_sampler="nutpie"):
        """
        Fit the data to Weibull distribution using MLE method.

        Returns
        -------
        trace : pymc3.MultiTrace
            The trace of the fitted Distribution.
        """

        with pm.Model() as model:
            # register the priors
            model.register_rv(self.coefficients, name="coeff")
            model.register_rv(self.shape, name="shape")

            # define the likelihood
            pm.Potential("llk", self._logp(self.lb, self.ub, self.event, self.rv))

            # sampling
            self.trace = pm.sample(draws=draw, tune=tune, chains=chains, cores=cores, nuts_sampler=nuts_sampler)

    def summary(self):
        """
        Print the summary of the fitted Weibull distribution.
        """
        summary = pd.DataFrame(az.summary(self.trace, round_to=3, kind="stats", hdi_prob=0.95))
        return summary.rename(columns={"hdi_2.5%": "BCI_lb", "hdi_97.5%": "BCI_ub"})

    def plot_trace(self, **kwargs):
        """
        Plot the trace of the fitted Weibull distribution.
        """
        return az.plot_trace(self.trace, **kwargs)

    def plot_hazard(self):
        """
        Plot the hazard function of the fitted Weibull distribution.
        """
        pass

    def plot_survival(self):
        """
        Plot the survival function of the fitted Weibull distribution.
        """
        pass

    def propotionality_test(self):
        """
        Perform the propotionality test of the covariates.
        """
        pass


class ExponentialPH(ProportionalHazard):
    """
    ExponentialPH class is used to fit the survival data to the exponential distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the ExperimentalPH class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        super().__init__(lb, ub, event, covariates, priors)
        self.scale = 1 / self.scale
        self.rv = pm.Exponential.dist(self.scale)

    def fit(self, draw=1000, tune=100, chains=4, cores=4, nuts_sampler="nutpie"):
        """
        Fit the data to Weibull distribution using MLE method.

        Returns
        -------
        trace : pymc3.MultiTrace
            The trace of the fitted Distribution.
        """

        with pm.Model() as model:
            # register the priors
            model.register_rv(self.coefficients, name="coeff")

            # define the likelihood
            pm.Potential("llk", self._logp(self.lb, self.ub, self.event, self.rv))

            # sampling
            self.trace = pm.sample(draws=draw, tune=tune, chains=chains, cores=cores, nuts_sampler=nuts_sampler)


class WeibullPH(ProportionalHazard):
    """
    WeibullPH class is used to fit the survival data to the Weibull distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the WeibullPH class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        super().__init__(lb, ub, event, covariates, priors)

        assert "shape" in priors, "shape prior is not given"
        self.shape = priors["shape"]
        self.rv = pm.Weibull.dist(self.shape, self.scale)


class GumbelPH(ProportionalHazard):
    """
    GumbelPH class is used to fit the survival data to the Gumbel distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the GumbelPH class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        super().__init__(lb, ub, event, covariates, priors)

        # check if the shape prior is given
        assert "shape" in priors, "shape prior is not given"
        self.shape = priors["shape"]
        self.rv = pm.Gumbel.dist(mu=self.scale, beta=self.shape)


class GammaPH(ProportionalHazard):
    """
    GammaPH class is used to fit the survival data to the Gamma distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the GammaPH class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        super().__init__(lb, ub, event, covariates, priors)

        # check if there has the censoring data, as the Gamma distribution is not suitable for the censored data
        if np.any(event != 0):
            warnings.warn("The Gamma distribution is not suitable for the censored data, "
                          "the censored data will be treated as fully observed data.")
            self.event = np.zeros_like(self.event)

        # check if the shape prior is given
        assert "shape" in priors, "shape prior is not given"
        self.shape = priors["shape"]
        self.rv = pm.Gamma.dist(alpha=self.shape, beta=1 / self.scale)

    def _logp(self, lb, ub, event, rv):
        """
        The log likelihood function of the Generalized Proportional Hazard model.
        :parameter
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        rv : PyMC3 distribution
            The random variable of the distribution
        """

        llk = (event == 0) * pm.logp(rv, lb)
        return llk


class GompertzPH(ProportionalHazard):
    """
    GompertzPH class is used to fit the survival data to the Gompertz distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the GompertzPH class with the data.

        Parameters
        ----------
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        covariates : array-like
        priors : dict
        """

        super().__init__(lb, ub, event, covariates, priors)

        # check if the shape prior is given
        assert "shape" in priors, "shape prior is not given"
        self.shape = priors["shape"]

    def _logp(self, lb, ub, event, rv):
        """
        The log likelihood function of the Generalized Proportional Hazard model.
        :parameter
        lb : array-like
            The left bound of the Time-to-Event data
        ub : array-like
            The right bound of the Time-to-Event data
        event : array-like
            The event indicator of the Time-to-Event data
            -1: Left censored
            0: fully observed
            1: Right censored
            2: Interval censored
        rv : PyMC3 distribution
            The random variable of the distribution
        """
        # PyMC has not implemented the Gompertz distribution
        # so we need to implement the log likelihood function manually
        assert self.rv is None

        def gompertz_logp(x, shape, scale):
            return pm.math.log(scale) + pm.math.log(shape) + shape + scale * x - pm.math.exp(scale * x) * shape

        def gompertz_logcdf(x, shape, scale):
            return pm.math.log(1 - pm.math.exp(- shape * (pm.math.exp(scale * x) - 1)))

        llk = (event == -1) * gompertz_logcdf(ub, self.shape, self.scale)
        llk += (event == 0) * gompertz_logp(lb, self.shape, self.scale)
        llk += (event == 1) * (1 - gompertz_logcdf(lb, self.shape, self.scale))
        llk += (event == 2) * (gompertz_logcdf(ub, self.shape, self.scale) -
                               gompertz_logcdf(lb, self.shape, self.scale))

        return llk
