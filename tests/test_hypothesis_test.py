import numpy as np
from hypervem.model import HypergraphCoClustering
from hypervem.hypothesis_test import goodness_of_fit_test, GoodnessOfFitResult
from hypervem._tracy_widom import tracy_widom_cdf, tracy_widom_ppf


def test_tracy_widom():
    crit = tracy_widom_ppf(0.95)
    assert np.isclose(crit, 0.9793, atol=1e-3)
    assert np.isclose(tracy_widom_cdf(crit), 0.95, atol=1e-3)


def test_goodness_of_fit():
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(30, 40))
    model = HypergraphCoClustering(n_node_clusters=2, n_hyperedge_clusters=2, max_iter=5, random_state=42)
    model.fit(X)
    res = goodness_of_fit_test(X, estimator=model)
    assert isinstance(res, GoodnessOfFitResult)
    assert 0.0 <= res.p_value <= 1.0
