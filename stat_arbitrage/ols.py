import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from pandas import Series, DataFrame

__all__ = ['RollingOLS']


def rwindows(a, window):
    """Create rolling window blocks from a given array.

    The shape of the result is meant to translate cleanly to pandas DataFrame
    convention of computing rolling statistics for blocks.

    Parameters
    ==========
    a : numpy.ndarray
        Of ndim {1, 2}
    window : int
        The window size

    Returns
    =======
    blocks : ndarray
        A higher-dimensional array containing each window (block)

    Shape of *a*            Shape of *blocks*
    ============            =================
    (x, )                   (x - window + 1, window, 1)
    (x, y)                  (x - window + 1, window, y)
    ...                     ...

    That is, each innermost element of the result is a window/block.
    """

    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = a.shape[0] - window + 1, window, a.shape[-1]
    strides = (a.strides[0],) + a.strides
    blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return blocks


class RollingOLS(object):
    """Provides rolling ordinary least squares (OLS) regression capability.

    Note: this approach is designed to be functional and user-friendly.  It
    works well on smaller (<10,000) datasets, but may create memory issues with
    datasets >100,000 samples.  It works by creating a RegressionWrapper for
    each rolling period, from which various regression attributes can be called.

    Parameters
    ==========
    endog : Series
        dependent variable
    exog : Series or DataFrame
        array of independent variable(s)
    window : int
        window length
    has_intercept : bool, default True
        if False, an intercept column equal to 1 will be added to exog
    """

    def __init__(self, endog, exog, window):
        self.endog = endog
        self.exog = exog
        self.window = window
        self._result_idx = self.exog.index[self.window - 1:]

        # Create a MultiIndex for 3-dimensional result data such as rolling
        #   residuals and fitted values.
        outer = np.repeat(self._result_idx.values, self.window, 0)
        inner = rwindows(self.exog.index.values, self.window).flatten()
        tups = list(zip(outer, inner))
        self._result_idx_3d = pd.MultiIndex.from_tuples(tups,
                                                        names=['Date Ending', 'Date'])

    def fit(self):
        """Container for RegressionResultsWrappers.

        Full regression results are ran once for each rolling window and
        stored where various attributes can later be called.
        """

        self.rendog = rwindows(self.endog.values, window=self.window)
        self.rexog = rwindows(self.exog.values, window=self.window)
        self.models = [smf.OLS(y, x, hasconst=True).fit() for y, x in
                       zip(self.rendog, self.rexog)]
        # return self to enable method chaining
        return self

    def _get(self, attr):
        """Call different regression attributes from statsmodels.OLS results.

        Internal method used to call @cache_readonly results from each
          RegressionResults wrapper.

        Available attributes are here:
        statsmodels.regression.linear_model.RegressionResults

        Parameters
        ==========
        attr : str
            string form of the attribute to call; example: 'tvalues'
        """

        return [getattr(n, attr) for n in self.models]

    # 1d data (return type is pd.Series)
    # These properties consist of a scalar for each rolling period.
    # --------------------------------------------------------------------------

    @property
    def aic(self):
        """Akaike information criterion."""
        return Series(self._get('aic'), index=self._result_idx,
                      name='aic')

    @property
    def bic(self):
        """Bayesian information criterion."""
        return Series(self._get('bic'), index=self._result_idx,
                      name='bic')

    @property
    def condition_number(self):
        """Return condition number of exogenous matrix.

        Calculated as ratio of largest to smallest eigenvalue.
        """
        return Series(self._get('condition_number'), index=self._result_idx,
                      name='condition_number')

    @property
    def df_model(self):
        """Model (regression) degrees of freedom (dof)."""
        return Series(self._get('df_model'), index=self._result_idx,
                      name='df_model')

    @property
    def df_resid(self):
        """Residual degrees of freedom (dof)."""
        return Series(self._get('df_resid'), index=self._result_idx,
                      name='df_resid')

    @property
    def df_total(self):
        """Total degrees of freedom (dof)."""
        return self.df_model + self.df_resid

    @property
    def ess(self):
        """Error sum of squares (sum of squared residuals)."""
        return Series(self._get('ess'), index=self._result_idx,
                      name='ess')

    @property
    def fstat(self):
        """F-statistic of the fully specified model.

        Calculated as the mean squared error of the model divided by the
        mean squared error of the residuals.
        """

        return Series(self._get('fvalue'), index=self._result_idx,
                      name='fstat')

    @property
    def f_pvalue(self):
        """p-value associated with the F-statistic."""
        return Series(self._get('f_pvalue'), index=self._result_idx,
                      name='f_pvalue')

    @property
    def mse_model(self):
        """Mean squared error of the model.

        The explained sum of squares  divided by the model dof.
        """

        return Series(self._get('mse_model'), index=self._result_idx,
                      name='mse_model')

    @property
    def mse_resid(self):
        """Mean squared error of the residuals.

        The sum of squared residuals divided by the residual dof.
        """

        return Series(self._get('mse_resid'), index=self._result_idx,
                      name='mse_resid')

    @property
    def mse_total(self):
        """Total mean squared error.

        The uncentered total sum of squares divided by nobs.
        """

        return Series(self._get('mse_total'), index=self._result_idx,
                      name='mse_total')

    @property
    def nobs(self):
        """Number of observations."""
        return Series(self._get('nobs'), index=self._result_idx,
                      name='nobs')

    @property
    def rss(self):
        """Regression sum of squares."""
        return Series(self._get('ssr'), index=self._result_idx,
                      name='rss')

    @property
    def rsq(self):
        """R-squared of a model with an intercept.

        This is defined here as 1 - ssr/centered_tss if the constant is
        included in the model and 1 - ssr/uncentered_tss if the constant is
        omitted.
        """
        return Series(self._get('rsquared'), index=self._result_idx,
                      name='rsq')

    @property
    def rsq_adj(self):
        """Adjusted R-squared of a model with an intercept.

        This is defined here as 1 - (nobs-1)/df_resid * (1-rsquared) if a
        constant is included and 1 - nobs/df_resid * (1-rsquared) if no
        constant is included.
        """
        return Series(self._get('rsquared_adj'), index=self._result_idx,
                      name='rsq_adj')

    @property
    def tss(self):
        """Total sum of squares."""
        return Series(self._get('centered_tss'), index=self._result_idx,
                      name='centered_tss')

    # 2d data (return type is pd.DataFrame)
    # For models with >1 exogenous variable, these properties consist of an
    #   nx1 vector for each rolling period.
    # --------------------------------------------------------------------------

    @property
    def coefs(self):
        """The linear coefficients that minimize the least squares criterion.

        This is usually called Beta for the classical linear model.
        """

        if isinstance(self.exog, DataFrame):
            return DataFrame(self._get('params'), index=self._result_idx,
                             columns=self.exog.columns)
        else:
            return pd.Series(self._get('params'), index=self._result_idx)

    @property
    def pvalues(self):
        """Returns the coefficient p-values in DataFrame form."""
        return DataFrame(self._get('pvalues'), index=self._result_idx,
                         columns=self.exog.columns)

    @property
    def tvalues(self):
        """Returns the coefficient t-statistics in DataFrame form."""
        return DataFrame(self._get('tvalues'), index=self._result_idx,
                         columns=self.exog.columns)

    @property
    def stderrs(self):
        """The standard errors of the parameter estimates."""
        return DataFrame(self._get('bse'), index=self._result_idx,
                         columns=self.exog.columns)

    # 3d data (return type is a MultiIndex pd.DataFrame)
    # Note that pd.Panel was deprecated in 0.20.1
    # For models with >1 exogenous variable, these properties consist of an
    #   nxm vector for each rolling period.
    # The "outer" index will be _result_idx (period-ending basis), with the
    #   inner indices being the individual periods within each outer period.
    # --------------------------------------------------------------------------

    @property
    def fitted_values(self):
        """The predicted the values for the original (unwhitened) design."""
        return Series(np.array(self._get('fittedvalues')).flatten(),
                      index=self._result_idx_3d,
                      name='fittedvalues')

    @property
    def resids(self):
        """The residuals of the model."""
        return Series(np.array(self._get('resid')).flatten(),
                      index=self._result_idx_3d,
                      name='resids')
