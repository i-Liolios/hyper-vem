import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
import warnings
from scipy import sparse
# Lazy import of KModes
try:
    from kmodes.kmodes import KModes
    HAS_KMODES = True
except ImportError:
    HAS_KMODES = False


def initialize_parameters(
    X: np.ndarray | sparse.csc_array, 
    K: int, 
    G: int, 
    method: str = "kmeans", 
    smooth: float = 1e-5,
    priors: tuple = None, 
    params: dict = None,
    random_state = None
):
    """
    Initializes parameters `gamma`, `delta`, `theta`, `c`, `d`.

    Methods
    -------
    - 'choose' : Uses user-specified parameters.
    - 'random' : Uses random initialization based on priors.
    - 'kmeans' : Uses K-Means clustering on rows/cols of X. (default)
    - 'kmodes' : Alternative initialization using K-modes for strictly binary data. This is
    much slower than any other initialization method. To be used as a fallback when
    K-means fails.

    Parameters
    ----------
    X : np.ndarray or sparse.csr_array
        The incidence matrix (M x N).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.

    method : str
        Initialization strategy.

    smooth : float
        Smoothing constant added to k-means/k-modes output.

    priors : tuple, optional
        (dirichlet_alpha, beta_alpha, beta_beta). Prior hyperparameters.
    
    params: dict, optional
        {'gamma': ..., 'delta': ..., 'theta': ...}. User provided parameter values. Required if
        method = "choose".

    Returns
    -------
    tuple
        (c, d, gamma, delta, theta)
        Note: gamma, delta, theta may be None if the method (like kmeans)
        defers their calculation to the first M-step.
    """
    N, M = X.shape[1], X.shape[0]

    rng = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
    if rng is None:
        rng = np.random.mtrand._rand

    # Default return values
    gamma, delta, theta = None, None, None
    c = np.ones((N, K)) / K
    d = np.ones((M, G)) / G

    if priors is None or any(p is None for p in priors):
        prior_dirichlet_alpha, prior_beta_alpha, prior_beta_beta = (1, 1, 1) # Flat priors
    else:   
        prior_dirichlet_alpha, prior_beta_alpha, prior_beta_beta = priors

    if method == "choose":
        if any(p is None for p in params):
            warnings.warn("gamma, delta and/or theta were not provided correctly. Reverting to random initialization")
            method = "random" 
        else:
            gamma = params.get("gamma")
            delta = params.get("delta")
            theta = params.get("theta")
            return c, d, gamma, delta, theta
            
    if method == "random":
        # All parameters are sampled from their priors
        gamma = rng.dirichlet([prior_dirichlet_alpha] * K)  
        delta = rng.dirichlet([prior_dirichlet_alpha] * G)  
        theta = rng.beta(prior_beta_alpha, prior_beta_beta, size=(K, G))
        d = rng.dirichlet([prior_dirichlet_alpha] * G, size=M)  
        c = rng.dirichlet([prior_dirichlet_alpha] * K, size=N)  
        return c, d, gamma, delta, theta

    if method == "kmeans":
        # Initialize Node responsibilities (c) using KMeans
        kmeans_nodes = KMeans(n_clusters=K, n_init=10, random_state=rng).fit(X.T)
        c = np.zeros((N, K))
        c[np.arange(N), kmeans_nodes.labels_] = 1.0
        # Add small noise to avoid log(0)
        c = (c + smooth) / (1 + smooth * K)

        # Initialize Hyperedge responsibilities (d) using KMeans
        kmeans_hyperedges = KMeans(n_clusters=G, n_init=10, random_state=rng).fit(X)
        d = np.zeros((M, G))
        d[np.arange(M), kmeans_hyperedges.labels_] = 1.0
        # Add small noise to avoid log(0)
        d = (d + smooth) / (1 + smooth * G)

        return c, d, gamma, delta, theta

    elif method == "kmodes":
        if not HAS_KMODES:
            raise ImportError("Package 'kmodes' not found. Cannot use method='kmodes'.")
        # Initialize Node responsibilities (c) using KModes
        kmodes_nodes = KModes(n_clusters=K, init="Huang", n_init=5, verbose=0, random_state=rng).fit(X.T)
        c = np.zeros((N, K))
        c[np.arange(N), kmodes_nodes.labels_] = 1.0
        c = (c + smooth) / (1 + smooth * K)
        # Initialize Hyperedge responsibilities (d) using KModes
        kmodes_hyperedges = KModes(n_clusters=G, init="Huang", n_init=5, verbose=0, random_state=rng).fit(X)   
        d = np.zeros((M, G))
        d[np.arange(M), kmodes_hyperedges.labels_] = 1.0
        d = (d + smooth) / (1 + smooth * G)

        return c, d, gamma, delta, theta


def get_prior_parameters(
    X: np.ndarray | sparse.csr_array,
    method: str | None = "weakly_informative", 
    mask: np.ndarray | None = None, 
    prior_params: dict | None = None
):
    """
    Determines the numerical values for Dirichlet and Beta priors based on the chosen method.

    Parameters
    ----------
    method : str
        The method used to set priors. Options:
        - 'flat': Sets all priors to 1.0.
        - 'jeffreys': Sets all priors to 0.5.
        - 'weakly_informative': Auto-scales Beta priors based on graph density. (default)
        - 'choose': Uses specific values provided in `current_params`.

    prior_params : dict
        Dictionary containing manually specified prior values. Used if method='choose'.
        Expected keys: 'dirichlet_alpha', 'beta_alpha', 'beta_beta'.

    X : np.ndarray or sparse.csr_array
        The incidence matrix.
    
    mask : np.ndarray
        The masking matrix, where NaNs -> 0 and NonNans -> 1.

    Returns
    -------
    tuple
        (dirichlet_alpha, beta_alpha, beta_beta)
    """

    if method == "choose":
        # Extract existing values or None
        dirichlet_alpha = prior_params.get("dirichlet_alpha")
        beta_alpha = prior_params.get("beta_alpha")
        beta_beta = prior_params.get("beta_beta")

        # If any value is None, warn and set to flat priors
        if dirichlet_alpha is None or beta_alpha is None or beta_beta is None:
            print("Warning: prior_params incomplete or empty. Using flat priors (=1.0) where missing.")
            dirichlet_alpha = dirichlet_alpha or 1.0
            beta_alpha = beta_alpha or 1.0
            beta_beta = beta_beta or 1.0
        return dirichlet_alpha, beta_alpha, beta_beta

    elif method == "flat":
        dirichlet_alpha, beta_alpha, beta_beta = 1.0, 1.0, 1.0
        return dirichlet_alpha, beta_alpha, beta_beta

    elif method == "jeffreys":
        dirichlet_alpha, beta_alpha, beta_beta = 0.5, 0.5, 0.5
        return dirichlet_alpha, beta_alpha, beta_beta

    elif method == "weakly_informative":
        # If no NaNs, calculate the mean
        # If NaNs present, sum mask to get the number of observations
        if mask is None:
            observed_density = X.mean()
        else:
            total_observed = np.sum(mask)
            observed_density = np.sum(X) / total_observed
        # Beta prior for connection probabilities: match mean to density
        beta_alpha = 1.0
        beta_beta = max(1.0, beta_alpha / observed_density - 1)
        # Dirichlet alpha 1.05: weakly informative, slight bias away from 0
        # to prevent empty clusters
        dirichlet_alpha = 1.05
        return dirichlet_alpha, beta_alpha, beta_beta

    else:
        warnings.warn("Invalid prior_method, returning flat priors.")
        dirichlet_alpha, beta_alpha, beta_beta = 1.0, 1.0, 1.0 
        return dirichlet_alpha, beta_alpha, beta_beta


def compute_responsibilities(
    X: np.ndarray | sparse.csr_array, 
    responsibilities: np.ndarray, 
    theta: np.ndarray, 
    mixing_proportions: np.ndarray, 
    mask: np.ndarray | None = None, 
    mode: str = "node", 
    eps: float = 1e-10
):
    r"""
    Computes node/hyperedge responsibilities (posteriors).

    Parameters
    ----------
    X : np.ndarray or sparse.csr_array
        The incidence matrix (M x N).

    responsibilities : np.ndarray
        The *other* responsibility matrix.
        - If computing 'c' (N, K), this is 'd' (M, G).
        - If computing 'd' (M, G), this is 'c' (N, K).

    theta : np.ndarray
        Connectivity matrix (K, G).

    mixing_proportions : np.ndarray
        Priors for the clusters being updated.
        - If computing 'c', this is 'gamma' (K,).
        - If computing 'd', this is 'delta' (G,).

    mask : np.ndarray, optional
        Mask for missing data.

    mode : str
        'node' or 'hyperedge'. Determines which dimension to solve for.

    Formulas
    --------

    **Node Update (mode='node')**
    
    The membership :math:`c_{ik}` is updated proportional to:

    .. math::
        c_{ik} \propto \gamma_k \prod_g \theta_{kg}^{Q_{ig}} (1-\theta_{kg})^{b_g - Q_{ig}}

    where the auxiliary variables are defined as:

    .. math::
        Q_{ig} = \sum_j X_{ji} d_{jg}, \quad b_g = \sum_j d_{jg}

    **Hyperedge Update (mode='hyperedge')**
    
    The membership :math:`d_{jg}` is updated proportional to:

    .. math::
        d_{jg} \propto \delta_g \prod_k \theta_{kg}^{Q_{jk}} (1-\theta_{kg})^{b_k - Q_{jk}}

    where the auxiliary variables are defined as:

    .. math::
        Q_{jk} = \sum_i X_{ji} c_{ik}, \quad b_k = \sum_i c_{ik}
        
    Returns
    -------
    np.ndarray
        The computed responsibility matrix ('c' or 'd').
    """
    nan_flag = mask is not None

    # Clip parameters to prevent log(0)
    theta_safe = np.clip(theta, eps, 1 - eps)
    mix_safe = np.clip(mixing_proportions, eps, 1 - eps)

    # Pre-compute logarithms
    log_theta = np.log(theta_safe)
    log_1_theta = np.log(1 - theta_safe)

    if mode == "node": # Compute c
        # Q[i, g] = Sum over j of (X_ji * d_jg)
        Q = X.T @ responsibilities

        # b[g] = Sum over j of d_jg
        if nan_flag:
            b = mask.T @ responsibilities
        else:
            b = responsibilities.sum(axis=0)

        # Term 1: Connections
        term1 = Q @ log_theta.T

        # Term 2: Non-connection
        term2 = (b - Q) @ log_1_theta.T

    elif mode == "hyperedge": # Compute d
        # Q[j, k] = Sum over i of (X_ji * c_ik)
        Q = X @ responsibilities

        # b[k] = Sum over i of c_ik
        if nan_flag:
            b = mask @ responsibilities
        else:
            b = responsibilities.sum(axis=0)

        # Term 1: Connections
        term1 = Q @ log_theta

        # Term 2: Non-connections
        term2 = (b - Q) @ log_1_theta

    else:
        raise ValueError("Mode must be 'node' or 'hyperedge'")

    logF = np.log(mix_safe)[None, :] + term1 + term2

    # Compute responsibilities and clip to prevent log(0)
    posterior = softmax(logF, axis=1)
    posterior = np.clip(posterior, eps, 1 - eps)

    # Normalize to maintain variational assumptions
    posterior /= posterior.sum(axis=1, keepdims=True)

    return posterior


def M_step(
    X: np.ndarray | sparse.csr_array, 
    c: np.ndarray, 
    d: np.ndarray, 
    mask: np.ndarray | None, 
    priors: tuple, 
    K: int, 
    G: int, 
    eps: float = 1e-10
):
    r"""
    Performs the M-step of the VEM algorithm.
    Updates `gamma`, `delta`, and `theta` based on `c` and `d`.

    Formulas
    --------
    
    .. math::

        \gamma_k &= \frac{\sum_i c_{ik} + \text{smooth}}{N + K \cdot \text{smooth}} \\
        \delta_g &= \frac{\sum_j d_{jg} + \text{smooth}}{M + G \cdot \text{smooth}} \\
        \theta_{kg} &= \frac{\sum_{j,i} d_{jg} X_{ji} c_{ik}}{(\sum_j d_{jg}) (\sum_i c_{ik})}

    where :math:`\text{smooth}` represents the additive smoothing factor.

    Parameters
    ----------
    X : np.ndarray or sparse.csr_array
        The incidence matrix.

    c : np.ndarray
        Node responsibilities (N, K).

    d : np.ndarray
        Hyperedge responsibilities (M, G).

    mask : np.ndarray
        Missing data mask.

    priors : tuple
        (dirichlet_alpha, beta_alpha, beta_beta).

    K : int
        Number of node clusters.

    G : int
        Number of hyperedge clusters.
    
    eps : float
        Stability constant to lower bound parameters. 

    Returns
    -------
    tuple
        (gamma, delta, theta)
    """
    prior_dirichlet_alpha, prior_beta_alpha, prior_beta_beta = priors
    N = X.shape[1]
    M = X.shape[0]

    nan_flag = mask is not None

    smooth_theta_denominator = prior_beta_alpha + prior_beta_beta - 2
    smooth_theta_numerator = prior_beta_alpha - 1
    smooth_delta = prior_dirichlet_alpha - 1
    smooth_gamma = prior_dirichlet_alpha - 1

    N_k = c.sum(axis=0)
    M_g = d.sum(axis=0)

    gamma = (N_k + smooth_gamma) / (N + K * smooth_gamma)
    delta = (M_g + smooth_delta) / (M + G * smooth_delta)

    # Numerator: (G, M) @ (M, N) @ (N, K) -> (G, K)
    numerator = d.T @ X @ c + smooth_theta_numerator

    # Denominator: Expected number of pairs (G, K)
    if nan_flag:
        denominator = d.T @ mask @ c + smooth_theta_denominator
    else:
        denominator = np.outer(M_g, N_k) + smooth_theta_denominator

    # theta_new: (K, G) -> Transpose result of (G, K) division
    theta = numerator.T / denominator.T

    # Clip parameters to avoid mode collapse
    gamma = np.clip(gamma, eps, 1 - eps)
    delta = np.clip(delta, eps, 1 - eps)
    theta = np.clip(theta, eps, 1 - eps)

    return gamma, delta, theta


def check_convergence(
    params_current: tuple = None, 
    params_old: tuple = None, 
    tol: float = 1e-4, 
    iteration: int = 0, 
    max_iter: int = 100,
    criterion: str = "parameters",
    elbo_current: float = None,
    elbo_old: float = None
):
    """
    Checks convergence of the VEM algorithm based on parameter changes or ELBO.

    Convergence is met either when the convergence metric (parameter change or 
    relative ELBO improvement) falls below the tolerance threshold, or when 
    the maximum number of iterations is reached.

    Parameters
    ----------
    params_current : tuple, optional
        A tuple containing the current parameters (theta, gamma, delta). 
        Required if criterion="parameters".

    params_old : tuple, optional
        A tuple containing the parameters from the previous iteration.
        Required if criterion="parameters".

    tol : float
        The tolerance threshold for convergence.

    iteration : int
        The current iteration number.

    max_iter : int
        The maximum number of allowed iterations.

    criterion : str
        The criterion to check for convergence. Options: "parameters", "elbo".

    elbo_current : float, optional
        The ELBO value of the current iteration. Required if criterion="elbo".

    elbo_old : float, optional
        The ELBO value of the previous iteration. Required if criterion="elbo".

    Returns
    -------
    bool
        True if the algorithm has converged or max_iter is reached, 
        False otherwise.
    """
    
    # Check max iterations first (applies to all criterions)
    if iteration >= max_iter:
        return True

    # Check convergence based on criterion
    if criterion == "parameters":
            
        theta, gamma, delta = params_current
        old_theta, old_gamma, old_delta = params_old

        delta_theta = np.linalg.norm(theta - old_theta)
        delta_gamma = np.linalg.norm(gamma - old_gamma)
        delta_delta = np.linalg.norm(delta - old_delta)

        max_change = max(delta_theta, delta_gamma, delta_delta)
        
        return max_change < tol

    elif criterion == "elbo":    
        # Avoid division by zero if elbo_old starts at 0
        if elbo_old == 0:
            rel_change = np.abs(elbo_current - elbo_old)
        else:
            rel_change = np.abs((elbo_current - elbo_old) / elbo_old)
        return rel_change < tol
