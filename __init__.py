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
        self.coefficients = priors["coeff"]

        # need to initialize by the child class
        self.scale = None
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

    def fit(self, draws=1000, tune=100, chains=4, cores=4, nuts_sampler="nutpie", **kwargs):
        """
        Fit the data to Weibull distribution using MLE method.

        Returns
        -------
        trace : pymc3.MultiTrace
            The trace of the fitted Distribution.
        """

        with pm.Model() as self.model:
            # register the priors
            self.model.register_rv(self.coefficients, name="coeff")
            self.model.register_rv(self.shape, name="shape")
            self.model.register_rv(self.scale, name="scale")

            # define the likelihood
            pm.Potential("logPdf", self._logp(self.lb, self.ub, self.event, self.rv))

            # sampling
            self.trace = pm.sample(draws=draws,
                                   tune=tune,
                                   chains=chains,
                                   cores=cores,
                                   nuts_sampler=nuts_sampler, **kwargs)

    def predict(self, covariates):
        """
        Predict the survival time of the given covariates.
        :param
            covariates: array-like
        :return:
            The predicted survival time.
        """
        raise NotImplementedError("The predict function is not implemented by the child class")

    def summary(self, **kwargs):
        """
        Print the summary of the fitted Weibull distribution.
        """
        summary = pd.DataFrame(az.summary(self.trace, round_to=3, kind="stats", hdi_prob=0.95, **kwargs))
        return summary.rename(columns={"hdi_2.5%": "BCI_lb", "hdi_97.5%": "BCI_ub"})

    def plot_trace(self, **kwargs):
        """
        Plot the trace of the fitted Weibull distribution.
        """
        return az.plot_trace(self.trace, **kwargs)

    def plot_posterior(self, **kwargs):
        """
        Plot the posterior of the fitted Weibull distribution.
        """
        return az.plot_posterior(self.trace, **kwargs)

    def plot_hazard(self):
        """
        Plot the hazard function of the fitted Weibull distribution.
        """
        raise NotImplementedError("The hazard function is not implemented by the child class")

    def plot_survival(self):
        """
        Plot the survival function of the fitted Weibull distribution.
        """
        raise NotImplementedError("The survival function is not implemented by the child class")

    def propotionality_test(self):
        """
        Perform the propotionality test of the covariates.
        """
        raise NotImplementedError("The propotionality test is not implemented by the child class")

    def visualize_model(self, **kwargs):
        """
        Visualize the model structure.
        """
        return pm.model_to_graphviz(self.model, **kwargs)


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
        self.shape = priors["shape"]
        self.scale = priors["scale"]

        self.rv = pm.Weibull.dist(
            self.shape, self.scale * pm.math.exp(-pm.math.dot(self.covariates, self.coefficients)) ** (1 / self.shape))


class ExponentialPH(ProportionalHazard):
    """
    ExponentialPH class is used to fit the survival data to the Exponential distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the ExponentialPH class with the data.

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
        self.scale = priors["scale"]

        assert "shape" not in priors, "The Exponential distribution does not have the shape parameter."

        self.rv = pm.Exponential.dist(
            self.scale * pm.math.exp(-pm.math.dot(self.covariates, self.coefficients)))


class LogNormalPH(ProportionalHazard):
    """
    LogNormalPH class is used to fit the survival data to the LogNormal distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the LogNormalPH class with the data.

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
        self.scale = priors["scale"]
        self.shape = priors["shape"]

        self.rv = pm.Lognormal.dist(
            self.scale * pm.math.exp(-pm.math.dot(self.covariates, self.coefficients)), self.shape)


class LogLogisticPH(ProportionalHazard):
    """
    LogLogisticPH class is used to fit the survival data to the LogLogistic distribution,
    while the Left , interval and right censored data are considered by using the MCMC method.
    """

    def __init__(self, lb, ub, event, covariates, priors):
        """
        Initialize the LogLogisticPH class with the data.

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
        self.scale = priors["scale"]
        self.shape = priors["shape"]

        self.rv = pm.Lognormal.dist(
            self.scale * pm.math.exp(-pm.math.dot(self.covariates, self.coefficients)), self.shape)
