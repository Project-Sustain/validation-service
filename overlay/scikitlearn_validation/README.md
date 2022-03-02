# ScikitLearn Model Attributes

## Regression

### Linear Regression

coef_ : array of shape (n_features, ) or (n_targets, n_features)
    Estimated coefficients for the linear regression problem.
    If multiple targets are passed during the fit (y 2D), this
    is a 2D array of shape (n_targets, n_features), while if only
    one target is passed, this is a 1D array of length n_features.

rank_ : int
    Rank of matrix `X`. Only available when `X` is dense.

singular_ : array of shape (min(X, y),)
    Singular values of `X`. Only available when `X` is dense.

intercept_ : float or array of shape (n_targets,)
    Independent term in the linear model. Set to 0.0 if
    `fit_intercept = False`.


### Gradient Boosting

feature_importances_ : ndarray of shape (n_features,)
    The impurity-based feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the (normalized)
    total reduction of the criterion brought by that feature.  It is also
    known as the Gini importance.

    Warning: impurity-based feature importances can be misleading for
    high cardinality features (many unique values). See
    :func:`sklearn.inspection.permutation_importance` as an alternative.

oob_improvement_ : ndarray of shape (n_estimators,)
    The improvement in loss (= deviance) on the out-of-bag samples
    relative to the previous iteration.
    ``oob_improvement_[0]`` is the improvement in
    loss of the first stage over the ``init`` estimator.
    Only available if ``subsample < 1.0``

train_score_ : ndarray of shape (n_estimators,)
    The i-th score ``train_score_[i]`` is the deviance (= loss) of the
    model at iteration ``i`` on the in-bag sample.
    If ``subsample == 1`` this is the deviance on the training data.

loss_ : LossFunction
    The concrete ``LossFunction`` object.

init_ : estimator
    The estimator that provides the initial predictions.
    Set via the ``init`` argument or ``loss.init_estimator``.

estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
    The collection of fitted sub-estimators.

n_classes_ : int
    The number of classes, set to 1 for regressors.

    .. deprecated:: 0.24
        Attribute ``n_classes_`` was deprecated in version 0.24 and
        will be removed in 1.1 (renaming of 0.26).

n_estimators_ : int
    The number of estimators as selected by early stopping (if
    ``n_iter_no_change`` is specified). Otherwise it is set to
    ``n_estimators``.

n_features_ : int
    The number of data features.

max_features_ : int
    The inferred value of max_features.


### Support Vector Regression

class_weight_ : ndarray of shape (n_classes,)
    Multipliers of parameter C for each class.
    Computed based on the ``class_weight`` parameter.

coef_ : ndarray of shape (1, n_features)
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

dual_coef_ : ndarray of shape (1, n_SV)
    Coefficients of the support vector in the decision function.

fit_status_ : int
    0 if correctly fitted, 1 otherwise (will raise warning)

intercept_ : ndarray of shape (1,)
    Constants in decision function.

n_support_ : ndarray of shape (n_classes,), dtype=int32
    Number of support vectors for each class.

shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
    Array dimensions of training vector ``X``.

support_ : ndarray of shape (n_SV,)
    Indices of support vectors.

support_vectors_ : ndarray of shape (n_SV, n_features)
    Support vectors.