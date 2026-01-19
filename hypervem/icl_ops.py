import numpy as np
from scipy.special import gammaln
from scipy import sparse


def compute_cdll(
    X: np.ndarray | sparse.csr_array,
    mask: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    theta: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """
    Computes the Expected Complete-Data Log-Likelihood (Expected CDLL) using matrix operations.
    
    This calculates E_q[ log P(X, Z, W) ], where the expectation is taken with 
    respect to the variational distributions `c` and `d`.

    Formula: E[Log-Priors(Z)] + E[Log-Priors(W)] + E[Log-Likelihood(X|Z,W)]

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_array
        Incidence matrix of shape (M, N).
    
    mask : np.ndarray or None
        Binary mask of shape (M, N) where 1 indicates observed and 0 indicates missing.
        If None, all entries are assumed observed.

    gamma : np.ndarray
        Node mixing proportions of shape (K,).
    
    delta : np.ndarray
        Hyperedge mixing proportions of shape (G,).
    
    theta : np.ndarray
        Bernoulli connectivity parameters of shape (K, G).
    
    c : np.ndarray
        Node variational responsibilities (soft assignments) of shape (N, K).
    
    d : np.ndarray
        Hyperedge variational responsibilities (soft assignments) of shape (M, G).

    Returns
    -------
    float
        The expected complete data log-likelihood.
    """

    # Latent Variable Priors
    prior_nodes = (c @ np.log(gamma)).sum()
    prior_hyperedges = (d @ np.log(delta)).sum()

    # Data Likelihood computation
    log_theta = np.log(theta)  
    log_1_minus_theta = np.log(1 - theta)  

    # Effective number of 'ones' in each block (k, g)
    eff_ones_block = (d.T @ X @ c).T

    # Total effective size of each block (k, g)
    col_sums_c = c.sum(axis=0)  
    row_sums_d = d.sum(axis=0)  

    if mask is None:
        eff_block_size = np.outer(col_sums_c, row_sums_d)  # (K, G)
    else:
        eff_block_size = (d.T @ mask @ c).T

    eff_zeros_block = eff_block_size - eff_ones_block

    data_likelihood = np.sum(
        eff_ones_block * log_theta + eff_zeros_block * log_1_minus_theta
    )

    return prior_nodes + prior_hyperedges + data_likelihood


def compute_entropy(variational_params: np.ndarray) -> float:
    """
    Entropy calculator for a variational distribution.

    Formula
    -------
    - Sum(p * log(p))

    Parameters
    ----------
    variational_params : np.ndarray
        Responsibility matrix (c or d).

    Returns
    -------
    float
        The entropy of the given distribution.
    """
    return -(variational_params * np.log(variational_params)).sum()


def compute_penalty(MN: int, K: int, G: int) -> float:
    """
    Computes the BIC penalty term based on model complexity.

    Parameters
    ----------
    MN : int
        Total number of observed entries (M * N or sum of mask).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    Returns
    -------
    float
        The penalty term for the given model complexity.
    """
    # Free parameters: Theta (K*G) + Gamma (K-1) + Delta (G-1)
    num_free_params = K * G + (K - 1) + (G - 1)

    penalty = 0.5 * num_free_params * np.log(MN)
    return penalty


def compute_exact_icl(
    X: np.ndarray | sparse.csr_array,
    mask: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    K: int,
    G: int,
    alpha_dirichlet: float,
    alpha_beta: float,
    beta_beta: float,
) -> float:
    """
    Computes the Exact Integrated Complete-data Likelihood (ICL) for the Hypergraph Co-clustering model.

    This replaces the BIC-like approximation (CDLL - Penalty) with an exact integration over
    the prior hyperparameters.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_array
        Incidence matrix (M x N).

    mask : np.ndarray or None
        Mask matrix (M x N) for missing data.

    c : np.ndarray
        Node responsibilities (N x K).

    d : np.ndarray
        Hyperedge responsibilities (M x G).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    alpha_dirichlet : float
        Alpha parameter of the Dirichlet prior.

    alpha_beta : float
        Alpha parameter of the Beta prior.

    beta_beta : float
        Beta parameter of the Beta prior.

    Returns
    -------
    float
        The exact ICL value.
    """
    M, N = X.shape

    # Create hard assignment matrices (one-hot encoding)
    c_hard = np.zeros_like(c)
    c_hard[np.arange(N), c.argmax(1)] = 1  

    d_hard = np.zeros_like(d)
    d_hard[np.arange(M), d.argmax(1)] = 1 

    # Number of elements in each row cluster g: (G,)
    n_g = d_hard.sum(axis=0)

    # Number of elements in each column cluster k: (K,)
    m_k = c_hard.sum(axis=0)

    # Number of 1s in each block (g, k): Result is (G, K)
    ones_count_block = (d_hard.T @ X @ c_hard).T

    # Total size of each block (k, g)
    if mask is None:
        total_count_block = np.outer(m_k, n_g)
    else:
        total_count_block = (d_hard.T @ mask @ c_hard).T

    zeros_count_block = total_count_block - ones_count_block

    # Dirichlet hyperparameters for node/hyperedge mixing proportions
    alpha = alpha_dirichlet
    # Beta hyperparameters for Bernoulli emission probabilities
    a, b = alpha_beta, beta_beta

    # Row Partitions P(Z) 
    term_row_partition = (
        gammaln(G * alpha)
        - gammaln(M + G * alpha)
        + np.sum(gammaln(n_g + alpha) - gammaln(alpha))
    )

    # Column Partitions P(W) 
    term_col_partition = (
        gammaln(K * alpha)
        - gammaln(N + K * alpha)
        + np.sum(gammaln(m_k + alpha) - gammaln(alpha))
    )

    # Sum over all blocks (k, g) of the Beta-Binomial integral
    term_data = (
        gammaln(a + b)
        - gammaln(total_count_block + a + b)
        + gammaln(ones_count_block + a)
        - gammaln(a)
        + gammaln(zeros_count_block + b)
        - gammaln(b)
    ).sum()

    return term_row_partition + term_col_partition + term_data


def compute_elbo(
    X: np.ndarray | sparse.csr_array,
    mask: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    theta: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """
    Computes the Evidence Lower Bound (ELBO) by summing CDLL and Entropies.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_array
        Incidence matrix (M x N).

    mask : np.ndarray or None
        Mask matrix (M x N).

    gamma : np.ndarray
        Node mixing proportions (K,).

    delta : np.ndarray
        Hyperedge mixing proportions (G,).

    theta : np.ndarray
        Block interaction probabilities (K x G).

    c : np.ndarray
        Node responsibilities (N x K).

    d : np.ndarray
        Hyperedge responsibilities (M x G).

    Returns
    -------
    float
        The computed ELBO value.
    """
    cdll = compute_cdll(X, mask, gamma, delta, theta, c, d)
    entropy_c = compute_entropy(c)
    entropy_d = compute_entropy(d)

    return cdll + entropy_c + entropy_d


def compute_icl(
    X: np.ndarray | sparse.csr_array,
    mask: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    K: int,
    G: int,
    alpha_dirichlet: float,
    alpha_beta: float,
    beta_beta: float,
    gamma: np.ndarray = None,
    delta: np.ndarray = None,
    theta: np.ndarray = None,
    approximation: bool = False,
) -> float:
    """
    Computes the Integrated Completed Likelihood (ICL).

    Delegates to either the exact computation or the BIC-like approximation.

    **Differences in Modes:**
    - **Approximation=False (Default):** Calculates the exact ICL using **hard** assignments (discretized `c` and `d`).
    - **Approximation=True:** Calculates a penalized ELBO using **soft** assignments. This serves as a proxy for ICL.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_array
        Incidence matrix (M x N).

    mask : np.ndarray or None
        Mask matrix (M x N).

    c : np.ndarray
        Node responsibilities (N x K).

    d : np.ndarray
        Hyperedge responsibilities (M x G).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    alpha_dirichlet : float
        Alpha parameter of Dirichlet prior.

    alpha_beta : float
        Alpha parameter of Beta prior.

    beta_beta : float
        Beta parameter of Beta prior.

    gamma : np.ndarray, optional
        Node mixing proportions (K,). Required if approximation=True.

    delta : np.ndarray, optional
        Hyperedge mixing proportions (G,). Required if approximation=True.

    theta : np.ndarray, optional
        Block interaction probabilities (K x G). Required if approximation=True.

    approximation : bool, optional
        Whether to use the BIC-like approximation (default=False).

    Returns
    -------
    float
        The computed ICL value.
    """
    if approximation:
        if gamma is None or delta is None or theta is None:
            raise ValueError(
                "Gamma, Delta, and Theta must be provided for ICL approximation."
            )

        elbo = compute_elbo(X, mask, gamma, delta, theta, c, d)

        if mask is None:
            MN = X.shape[0] * X.shape[1]
        else:
            MN = np.sum(mask)
        penalty = compute_penalty(MN, K, G)
        return elbo - penalty

    else:
        return compute_exact_icl(
            X, mask, c, d, K, G, alpha_dirichlet, alpha_beta, beta_beta
        )
