import pandas as pd
from sklearn import linear_model


def rolling_beta(X, y, idx, window=100):
    assert len(X) == len(y)

    out_dates = []
    out_beta = []

    model_ols = linear_model.LinearRegression()

    for iStart in range(0, len(X) - window):
        iEnd = iStart + window

        _x = X[iStart:iEnd].values.reshape(-1, 1)
        _y = y[iStart:iEnd].values.reshape(-1, 1)

        model_ols.fit(_x, _y)

        # store output
        out_dates.append(idx[iEnd])
        out_beta.append(model_ols.coef_[0][0])

    return pd.DataFrame({'beta': out_beta}, index=out_dates)


