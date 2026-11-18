import numpy as np
from scipy.special import softmax, gammaln
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


def _block_counts(X_cats, c, d):
    """
    Computes expected category counts N_kgc of shape (K, G, C).
    """
    K, G, C = c.shape[1], d.shape[1], len(X_cats)
    N_kgc = np.zeros((K, G, C), dtype=float)
    for c_idx, X_c in enumerate(X_cats):
        val = d.T @ X_c @ c
        N_kgc[:, :, c_idx] = val.toarray().T if sparse.issparse(val) else val.T
    return N_kgc


def prepare_categorical_matrices(X: np.ndarray | sparse.csr_array, n_categories: int | None = None):
    """
    Splits a categorical hypergraph incidence matrix X into per-category indicator matrices.

    Parameters
    ----------
    X : Hypergraph, array-like, or scipy.sparse matrix of shape (M, N)
        Discrete incidence matrix (M hyperedges x N nodes) with entries in
        {0, ..., C-1}. Missing entries should be represented as np.nan.

    n_categories : int, optional
        Total number of categories C. If None, inferred as max(X) + 1.

    Returns
    -------
    tuple
        (X_cats, mask, categories, C) where X_cats is a list of C sparse
        csr_array indicators of shape (M, N), and mask is the observed boolean mask.
    """
    if hasattr(X, "incidence_matrix"):
        X = X.incidence_matrix

    X_arr = X.toarray().astype(float) if sparse.issparse(X) else np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"Input X must be a 2D incidence matrix, got shape {X_arr.shape}.")

    mask = ~np.isnan(X_arr)
    C = int(np.nanmax(X_arr)) + 1 if n_categories is None else int(n_categories)
    categories = np.arange(C)
    X_cats = [sparse.csr_array(X_arr == c, dtype=float) for c in range(C)]

    return X_cats, mask, categories, C


def initialize_categorical_parameters(
    X_cats: list,
    mask: np.ndarray,
    K: int,
    G: int,
    C: int,
    method: str = "kmeans",
    smooth: float = 1e-5,
    params: dict | None = None,
    random_state=None,
):
    """
    Initializes parameters `gamma`, `delta`, `pi`, `c`, `d` for Categorical VEM.

    Methods
    -------
    - 'choose' : Uses user-specified parameters.
    - 'random' : Samples initial responsibilities and parameters from Dirichlet priors.
    - 'kmeans' : Uses K-Means clustering on one-hot encoded hyperedges/nodes of X. (default)

    Parameters
    ----------
    X_cats : list of scipy.sparse.csr_array
        List of C category indicator matrices of shape (M, N).

    mask : np.ndarray
        Boolean observation mask of shape (M, N).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    C : int
        Number of categories.

    method : str, optional (default='kmeans')
        Initialization strategy.

    smooth : float, optional (default=1e-5)
        Smoothing constant added to k-means assignments to avoid log(0).

    params : dict, optional
        {'gamma': ..., 'delta': ..., 'pi': ...} used when method='choose'.

    random_state : int, RandomState instance or None
        Random seed or generator.

    Returns
    -------
    tuple
        (c, d, gamma, delta, pi)
        Note: gamma, delta, pi are None when method='kmeans' (deferred to M-step).
    """
    M, N = X_cats[0].shape
    rng = check_random_state(random_state)

    c = np.ones((N, K)) / K
    d = np.ones((M, G)) / G

    if method == "choose" and params is not None:
        return c, d, params.get("gamma"), params.get("delta"), params.get("pi")

    if method == "random":
        c = rng.dirichlet(np.ones(K) * 1.5, size=N)
        d = rng.dirichlet(np.ones(G) * 1.5, size=M)
        gamma = np.ones(K) / K
        delta = np.ones(G) / G
        pi = rng.dirichlet(np.ones(C), size=(K, G))
        return c, d, gamma, delta, pi

    if method == "kmeans":
        feats_hyperedges = [
            np.where(mask, mat.toarray() if sparse.issparse(mat) else mat, 1.0 / C)
            for mat in X_cats
        ]
        hyperedge_features = np.hstack(feats_hyperedges)
        node_features = np.hstack([f.T for f in feats_hyperedges])

        kmeans_nodes = KMeans(n_clusters=K, n_init=10, random_state=rng).fit(node_features)
        c = np.zeros((N, K))
        c[np.arange(N), kmeans_nodes.labels_] = 1.0
        c = (c + smooth) / (1.0 + smooth * K)

        kmeans_edges = KMeans(n_clusters=G, n_init=10, random_state=rng).fit(hyperedge_features)
        d = np.zeros((M, G))
        d[np.arange(M), kmeans_edges.labels_] = 1.0
        d = (d + smooth) / (1.0 + smooth * G)

        return c, d, None, None, None

    raise ValueError(f"Unknown initialization method: {method}")


def compute_categorical_responsibilities(
    X_cats: list,
    responsibilities: np.ndarray,
    pi: np.ndarray,
    mixing_proportions: np.ndarray,
    mode: str = "node",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes node/hyperedge responsibilities (posteriors) for Categorical Hypergraph Co-clustering.

    Parameters
    ----------
    X_cats : list of scipy.sparse.csr_array
        List of C category indicator matrices of shape (M, N).

    responsibilities : np.ndarray
        The *other* responsibility matrix:
        - If computing 'c' (N, K), this is 'd' (M, G).
        - If computing 'd' (M, G), this is 'c' (N, K).

    pi : np.ndarray
        Categorical connection probability tensor of shape (K, G, C).

    mixing_proportions : np.ndarray
        Cluster mixing weights ('gamma' for mode='node', 'delta' for mode='hyperedge').

    mode : str, optional (default='node')
        'node' or 'hyperedge'.

    eps : float, optional (default=1e-12)
        Clipping threshold to avoid log(0).

    Returns
    -------
    np.ndarray
        Updated responsibility matrix ('c' or 'd').
    """
    if mode not in ("node", "hyperedge"):
        raise ValueError("Mode must be 'node' or 'hyperedge'")

    pi_safe = np.clip(pi, eps, 1.0)
    pi_safe /= pi_safe.sum(axis=-1, keepdims=True)
    log_pi = np.log(pi_safe)

    mix_safe = np.clip(mixing_proportions, eps, 1.0)
    log_mix = np.log(mix_safe / mix_safe.sum())

    if mode == "node":
        term_ll = np.zeros((X_cats[0].shape[1], log_pi.shape[0]), dtype=float)
    else:
        term_ll = np.zeros((X_cats[0].shape[0], log_pi.shape[1]), dtype=float)

    for c_idx, X_c in enumerate(X_cats):
        if mode == "node":
            Q_c = X_c.T @ responsibilities
            log_pi_c = log_pi[:, :, c_idx].T
        else:
            Q_c = X_c @ responsibilities
            log_pi_c = log_pi[:, :, c_idx]
        if sparse.issparse(Q_c):
            Q_c = Q_c.toarray()
        term_ll += Q_c @ log_pi_c

    logF = log_mix[None, :] + term_ll
    posterior = softmax(logF, axis=1)
    posterior = np.clip(posterior, eps, 1.0 - eps)
    return posterior / posterior.sum(axis=1, keepdims=True)


def categorical_m_step(
    X_cats: list,
    c: np.ndarray,
    d: np.ndarray,
    K: int,
    G: int,
    C: int,
    alpha_dirichlet_mix: float = 1.0,
    alpha_dirichlet_pi: float = 1.0,
    N_kgc: np.ndarray | None = None,
    eps: float = 1e-12,
):
    """
    Performs the M-step of the Categorical VEM algorithm.
    Updates `gamma`, `delta`, and `pi` based on `c` and `d`.

    Parameters
    ----------
    X_cats : list of scipy.sparse.csr_array
        Category indicator matrices (M x N).

    c : np.ndarray
        Node responsibilities (N, K).

    d : np.ndarray
        Hyperedge responsibilities (M, G).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    C : int
        Number of categories.

    alpha_dirichlet_mix : float, optional (default=1.0)
        Dirichlet hyperparameter for mixing weights.

    alpha_dirichlet_pi : float, optional (default=1.0)
        Dirichlet hyperparameter for cluster-pair category distributions.

    N_kgc : np.ndarray, optional
        Precomputed expected category counts of shape (K, G, C).

    eps : float, optional (default=1e-12)
        Stability constant to lower bound parameters.

    Returns
    -------
    tuple
        (gamma, delta, pi)
    """
    N, M = c.shape[0], d.shape[0]
    smooth_mix = max(0.0, alpha_dirichlet_mix - 1.0)
    gamma = (c.sum(axis=0) + smooth_mix) / (N + K * smooth_mix)
    delta = (d.sum(axis=0) + smooth_mix) / (M + G * smooth_mix)

    if N_kgc is None:
        N_kgc = _block_counts(X_cats, c, d)

    smooth_pi = max(0.0, alpha_dirichlet_pi - 1.0)
    denom = np.maximum(N_kgc.sum(axis=-1, keepdims=True) + C * smooth_pi, eps)
    pi = (N_kgc + smooth_pi) / denom

    gamma = np.clip(gamma, eps, 1.0 - eps)
    gamma /= gamma.sum()
    delta = np.clip(delta, eps, 1.0 - eps)
    delta /= delta.sum()

    pi = np.clip(pi, eps, 1.0)
    pi /= pi.sum(axis=-1, keepdims=True)
    return gamma, delta, pi


def compute_categorical_cdll(
    X_cats: list,
    gamma: np.ndarray,
    delta: np.ndarray,
    pi: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    N_kgc: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Computes the Expected Complete-Data Log-Likelihood (Expected CDLL).
    """
    log_pi = np.log(np.clip(pi, eps, 1.0))
    prior_nodes = (c @ np.log(np.clip(gamma, eps, 1.0))).sum()
    prior_hyperedges = (d @ np.log(np.clip(delta, eps, 1.0))).sum()

    if N_kgc is None:
        N_kgc = _block_counts(X_cats, c, d)

    data_ll = np.sum(N_kgc * log_pi)
    return float(prior_nodes + prior_hyperedges + data_ll)


def compute_categorical_elbo(
    X_cats: list,
    gamma: np.ndarray,
    delta: np.ndarray,
    pi: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    N_kgc: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Computes the Evidence Lower Bound (ELBO) by summing Expected CDLL and Entropies.
    """
    cdll = compute_categorical_cdll(X_cats, gamma, delta, pi, c, d, N_kgc=N_kgc, eps=eps)
    ent_c = float(-(c * np.log(np.clip(c, eps, 1.0))).sum())
    ent_d = float(-(d * np.log(np.clip(d, eps, 1.0))).sum())
    return cdll + ent_c + ent_d


def compute_categorical_icl(
    X_cats: list,
    c: np.ndarray,
    d: np.ndarray,
    K: int,
    G: int,
    C: int,
    gamma: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    pi: np.ndarray | None = None,
    alpha_dirichlet: float = 1.0,
    approximation: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Computes the Integrated Completed Likelihood (ICL) for Categorical Hypergraph Co-clustering.

    Parameters
    ----------
    X_cats : list of scipy.sparse.csr_array
        Category indicator matrices (M x N).

    c : np.ndarray
        Node responsibilities (N x K).

    d : np.ndarray
        Hyperedge responsibilities (M x G).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    C : int
        Number of categories.

    gamma : np.ndarray, optional
        Node mixing proportions (K,). Required if approximation=True.

    delta : np.ndarray, optional
        Hyperedge mixing proportions (G,). Required if approximation=True.

    pi : np.ndarray, optional
        Categorical interaction probabilities (K, G, C). Required if approximation=True.

    alpha_dirichlet : float, optional (default=1.0)
        Symmetric Dirichlet prior hyperparameter.

    approximation : bool, optional (default=False)
        If True, returns BIC-like penalized Expected CDLL; if False, computes exact integrated ICL.

    eps : float, optional (default=1e-12)
        Clipping threshold for approximate likelihood.

    Returns
    -------
    float
        The computed ICL value.
    """
    if approximation:
        if gamma is None or delta is None or pi is None:
            raise ValueError("gamma, delta, and pi must be provided for ICL approximation.")
        N_kgc = _block_counts(X_cats, c, d)
        cdll = compute_categorical_cdll(X_cats, gamma, delta, pi, c, d, N_kgc=N_kgc, eps=eps)
        nu = K * G * (C - 1) + (K - 1) + (G - 1)
        return float(cdll - 0.5 * nu * np.log(max(1.0, float(N_kgc.sum()))))

    N, M = c.shape[0], d.shape[0]
    c_hard = np.zeros_like(c)
    c_hard[np.arange(N), c.argmax(1)] = 1.0

    d_hard = np.zeros_like(d)
    d_hard[np.arange(M), d.argmax(1)] = 1.0

    N_k = c_hard.sum(axis=0)
    M_g = d_hard.sum(axis=0)
    N_kgc = _block_counts(X_cats, c_hard, d_hard)

    # Node and hyperedge partition integrals P(Z) and P(W)
    prior_gamma = (
        gammaln(K * alpha_dirichlet)
        - gammaln(N + K * alpha_dirichlet)
        + np.sum(gammaln(N_k + alpha_dirichlet) - gammaln(alpha_dirichlet))
    )
    prior_delta = (
        gammaln(G * alpha_dirichlet)
        - gammaln(M + G * alpha_dirichlet)
        + np.sum(gammaln(M_g + alpha_dirichlet) - gammaln(alpha_dirichlet))
    )

    # Dirichlet-Multinomial marginal integral across cluster pairs (K, G)
    sum_alpha_c = C * alpha_dirichlet
    log_constant = gammaln(sum_alpha_c) - C * gammaln(alpha_dirichlet)
    cocluster_marginals = (
        log_constant
        + gammaln(N_kgc + alpha_dirichlet).sum(axis=-1)
        - gammaln(N_kgc.sum(axis=-1) + sum_alpha_c)
    ).sum()

    return float(prior_gamma + prior_delta + cocluster_marginals)
