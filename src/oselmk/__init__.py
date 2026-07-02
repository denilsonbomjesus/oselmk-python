"""oselmk — Online Sequential ELM with Kernels.

Public API
----------
The top-level package exposes the main model class and the package
version.  Everything else lives under :mod:`oselmk.utils`.

Quick-start
-----------
::

    from oselmk import OSELMK
    from oselmk.utils import ZScoreNormalizer, make_lag_features, compute_all

    # 1. Build lag features from a univariate series
    X, y = make_lag_features(series, n_lags=4)

    # 2. Normalise (fit on training set only)
    scaler_X = ZScoreNormalizer()
    X_train_n = scaler_X.fit_transform(X_train)
    X_test_n  = scaler_X.transform(X_test)

    scaler_y = ZScoreNormalizer()
    y_train_n = scaler_y.fit_transform(y_train)

    # 3. Offline batch fit
    model = OSELMK(C=100.0, kernel="rbf")
    model.fit(X_train_n, y_train_n)

    # 4. Online sequential update
    model.update(X_new_n, y_new_n, mode="sequential")

    # 5. Predict and denormalise
    y_pred = scaler_y.inverse_transform(model.predict(X_test_n))

    # 6. Evaluate
    metrics = compute_all(y_test, y_pred)

Reference
---------
Huang, G., et al. (2014). "Online sequential extreme learning machine with
kernels for nonstationary time series prediction." *Neurocomputing*, 145,
90–97. https://doi.org/10.1016/j.neucom.2014.05.068
"""

from __future__ import annotations

from oselmk.model import OSELMK

__version__: str = "0.1.0"

__all__: list[str] = [
    # Core model
    "OSELMK",
    # Package metadata
    "__version__",
]
