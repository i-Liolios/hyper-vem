import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV

from hypervem import CategoricalCoClustering, HypergraphCoClustering
from hypervem.categorical_vem_ops import (
    compute_categorical_icl,
    prepare_categorical_matrices,
)
from hypervem.hypergraph import Hypergraph


def _make_categorical_hypergraph(M=100, N=80, K=3, G=3, C=3, seed=42):
    rng = np.random.RandomState(seed)
    true_hyperedges = rng.choice(G, size=M)
    true_nodes = rng.choice(K, size=N)

    true_pi = np.full((K, G, C), 0.1)
    for k in range(K):
        for g in range(G):
            true_pi[k, g, (k + g) % C] += 0.7
    true_pi /= true_pi.sum(axis=-1, keepdims=True)

    X = np.empty((M, N), dtype=float)
    for j in range(M):
        for i in range(N):
            X[j, i] = rng.choice(C, p=true_pi[true_nodes[i], true_hyperedges[j]])
    return X, true_hyperedges, true_nodes


def test_prepare_categorical_matrices():
    X = np.array([[0, 1, 2], [2, np.nan, 0]])
    X_cats, mask, _, C = prepare_categorical_matrices(X)
    assert C == 3 and len(X_cats) == 3
    assert mask[0, 0] and not mask[1, 1]
    assert X_cats[0][0, 0] == 1.0 and X_cats[2][1, 0] == 1.0 and X_cats[1][1, 1] == 0.0

    X_sp = sparse.csr_matrix([[0, 1], [2, 0]])
    X_cats_sp, _, _, C_sp = prepare_categorical_matrices(X_sp)
    assert C_sp == 3 and X_cats_sp[0][0, 0] == 1.0 and X_cats_sp[2][1, 0] == 1.0


def test_synthetic_recovery_and_sklearn_api():
    X, true_hyperedges, true_nodes = _make_categorical_hypergraph(M=120, N=90, K=3, G=3, C=3, seed=123)
    hg = Hypergraph(X)
    model = CategoricalCoClustering(
        n_node_clusters=3, n_hyperedge_clusters=3, n_init=3, max_iter=60, random_state=42
    )
    hyperedge_preds = model.fit_predict(hg)
    assert hyperedge_preds.shape == (120,)
    assert np.array_equal(hyperedge_preds, model.labels_)
    assert np.array_equal(model.hyperedge_clusters_, model.hyperedge_labels_)
    assert np.array_equal(model.node_clusters_, model.node_labels_)
    assert adjusted_rand_score(true_hyperedges, model.hyperedge_clusters_) > 0.85
    assert adjusted_rand_score(true_nodes, model.node_clusters_) > 0.85

    he_mask, node_mask = model.coclusters_
    assert he_mask.shape == (3, 120) and node_mask.shape == (3, 90)
    np.testing.assert_allclose(model.node_responsibilities_.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(model.hyperedge_responsibilities_.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(model.block_probs_.sum(axis=-1), 1.0, atol=1e-5)
    assert model.predict_proba(X[:5]).shape == (5, 3)


def test_elbo_monotonicity_and_missing_data():
    X, true_hyperedges, true_nodes = _make_categorical_hypergraph(M=100, N=80, K=2, G=2, C=3, seed=42)
    rng = np.random.RandomState(42)
    X_missing = X.copy()
    X_missing[rng.rand(*X.shape) < 0.25] = np.nan

    model = CategoricalCoClustering(
        n_node_clusters=2, n_hyperedge_clusters=2, n_init=3, max_iter=50, random_state=42
    ).fit(X_missing)

    assert len(model.elbo_history_) >= 2
    assert np.all(np.diff(model.elbo_history_) >= -1e-4)
    assert adjusted_rand_score(true_hyperedges, model.hyperedge_clusters_) > 0.80
    assert adjusted_rand_score(true_nodes, model.node_clusters_) > 0.80


def test_exact_icl_uses_hard_assignments_and_matches_binary():
    rng = np.random.RandomState(7)
    X_bin = rng.randint(0, 2, size=(30, 25)).astype(float)
    X_cats, _, _, C = prepare_categorical_matrices(X_bin, n_categories=2)

    c_soft = np.array([[0.8, 0.2]] * 15 + [[0.3, 0.7]] * 10)
    d_soft = np.array([[0.9, 0.1]] * 15 + [[0.2, 0.8]] * 15)
    c_hard = (c_soft == c_soft.max(axis=1, keepdims=True)).astype(float)
    d_hard = (d_soft == d_soft.max(axis=1, keepdims=True)).astype(float)

    assert np.isclose(
        compute_categorical_icl(X_cats, c_soft, d_soft, 2, 2, C),
        compute_categorical_icl(X_cats, c_hard, d_hard, 2, 2, C),
    )

    bin_mod = HypergraphCoClustering(
        n_node_clusters=2, n_hyperedge_clusters=2, prior_method="flat", n_init=1, random_state=7
    ).fit(X_bin)
    cat_mod = CategoricalCoClustering(
        n_node_clusters=2, n_hyperedge_clusters=2, n_categories=2, n_init=1, random_state=7
    ).fit(X_bin)
    cat_icl_on_bin_resp = compute_categorical_icl(
        X_cats, bin_mod.node_responsibilities_, bin_mod.hyperedge_responsibilities_, 2, 2, 2, alpha_dirichlet=1.0
    )
    assert np.isclose(cat_icl_on_bin_resp, bin_mod.compute_ICL(approximation=False))
    assert np.isfinite(cat_mod.score(X_bin))


def test_gridsearch_compatibility():
    X, _, _ = _make_categorical_hypergraph(M=40, N=30, K=2, G=2, C=3, seed=1)
    idx = np.arange(X.shape[0])
    gs = GridSearchCV(
        CategoricalCoClustering(n_categories=3, n_init=2, max_iter=30, random_state=1),
        param_grid={"n_node_clusters": [2], "n_hyperedge_clusters": [2, 3]},
        cv=[(idx, idx)],
    ).fit(X)
    assert gs.best_params_ == {"n_node_clusters": 2, "n_hyperedge_clusters": 2}
