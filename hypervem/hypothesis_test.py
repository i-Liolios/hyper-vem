from dataclasses import dataclass
import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import svdvals
from scipy.sparse.linalg import svds
from sklearn.utils.validation import check_array, check_is_fitted

from ._tracy_widom import tracy_widom_ppf, tracy_widom_cdf


@dataclass
class GoodnessOfFitResult:
    statistic: float
    p_value: float
    critical_value: float
    reject_null: bool
    lambda_max: float
    centering_a: float
    scaling_b: float
    n_row_clusters: int
    n_col_clusters: int
    alpha: float
    diagnostics: dict


def goodness_of_fit_test(
    X,
    row_labels=None,
    col_labels=None,
    estimator=None,
    alpha=0.05,
    binary=None,
):
    """
    Goodness-of-fit hypothesis test for latent block models.

    Parameters
    ----------
    X : Hypergraph, np.ndarray, or sparse matrix
        Input data matrix or Hypergraph instance.
    row_labels : array-like, optional
        Row cluster assignments.
    col_labels : array-like, optional
        Column cluster assignments.
    estimator : HypergraphCoClustering, optional
        Fitted co-clustering model.
    alpha : float, default=0.05
        Significance level.
    binary : bool, optional
        Whether data is binary. If None, auto-detected.

    Returns
    -------
    GoodnessOfFitResult
    """
    if hasattr(X, "incidence_matrix"):
        X = X.incidence_matrix

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if estimator is not None:
        check_is_fitted(estimator, ["hyperedge_clusters_", "node_clusters_"])
        row_labels = estimator.hyperedge_clusters_
        col_labels = estimator.node_clusters_
        G0 = getattr(estimator, "n_hyperedge_clusters", None)
        K0 = getattr(estimator, "n_node_clusters", None)
    elif row_labels is None or col_labels is None:
        raise ValueError("Must provide either a fitted estimator or row_labels and col_labels.")
    else:
        G0 = None
        K0 = None

    if sparse.issparse(X):
        X_mat = X.tocsr()
        has_nan = np.isnan(X_mat.data).any()
    else:
        X_mat = check_array(X, ensure_2d=True, ensure_all_finite=False)
        has_nan = np.isnan(X_mat).any()

    if has_nan:
        raise ValueError("goodness_of_fit_test does not support missing values (NaNs).")

    M, N = X_mat.shape

    row_labels = np.asarray(row_labels)
    col_labels = np.asarray(col_labels)
    if len(row_labels) != M or len(col_labels) != N:
        raise ValueError(f"Label shapes ({len(row_labels)}, {len(col_labels)}) do not match matrix ({M}, {N}).")

    row_u, row_inv = np.unique(row_labels, return_inverse=True)
    col_u, col_inv = np.unique(col_labels, return_inverse=True)
    effective_G = len(row_u)
    effective_K = len(col_u)

    if G0 is None:
        G0 = effective_G
    if K0 is None:
        K0 = effective_K

    row_counts = np.bincount(row_inv, minlength=effective_G)
    col_counts = np.bincount(col_inv, minlength=effective_K)

    if binary is None:
        data = X_mat.data if sparse.issparse(X_mat) else X_mat
        binary = bool(np.all((data == 0) | (data == 1)))

    R = sparse.csr_matrix((np.ones(M), (np.arange(M), row_inv)), shape=(M, effective_G))
    C = sparse.csr_matrix((np.ones(N), (np.arange(N), col_inv)), shape=(N, effective_K))
    block_sizes = np.outer(row_counts, col_counts)

    block_sums = (R.T @ X_mat @ C).toarray() if sparse.issparse(X_mat) else (R.T @ X_mat @ C)
    B_hat = block_sums / block_sizes

    X_dense = X_mat.toarray().astype(float) if sparse.issparse(X_mat) else np.asarray(X_mat, dtype=float)
    P_hat = B_hat[row_inv[:, None], col_inv[None, :]]

    if binary:
        var_hat = np.maximum(0.0, B_hat * (1.0 - B_hat))
    else:
        diff_sq = (X_dense - P_hat) ** 2
        var_hat = np.maximum(0.0, (R.T @ diff_sq @ C) / block_sizes)

    S_hat = np.sqrt(var_hat)
    sigma_hat = S_hat[row_inv[:, None], col_inv[None, :]]
    Z = np.divide(X_dense - P_hat, sigma_hat, out=np.zeros_like(X_dense, dtype=float), where=sigma_hat > 1e-12)

    n_zero_blocks = int(np.sum(S_hat <= 1e-12))
    if n_zero_blocks > 0:
        warnings.warn(f"{n_zero_blocks} block(s) have zero variance; residuals set to 0.", UserWarning)

    s1 = svdvals(Z)[0] if min(M, N) <= 500 else svds(Z, k=1, return_singular_vectors=False)[0]
    lambda_max = s1 ** 2

    sqrt_M, sqrt_N = np.sqrt(M), np.sqrt(N)
    a = (sqrt_M + sqrt_N) ** 2
    b = (sqrt_M + sqrt_N) * (1.0 / sqrt_M + 1.0 / sqrt_N) ** (1.0 / 3.0)

    T = (lambda_max - a) / b
    crit_val = tracy_widom_ppf(1.0 - alpha)
    p_val = 1.0 - tracy_widom_cdf(T)

    diagnostics = {
        "zero_variance_blocks": n_zero_blocks,
        "block_means": B_hat,
        "block_stds": S_hat,
    }

    return GoodnessOfFitResult(
        statistic=T,
        p_value=p_val,
        critical_value=crit_val,
        reject_null=T >= crit_val,
        lambda_max=lambda_max,
        centering_a=a,
        scaling_b=b,
        n_row_clusters=G0,
        n_col_clusters=K0,
        alpha=alpha,
        diagnostics=diagnostics,
    )
