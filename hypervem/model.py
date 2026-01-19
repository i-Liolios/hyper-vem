import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClusterMixin 
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils import check_random_state
from scipy import sparse

from . import icl_ops
from .hypergraph import Hypergraph
from . import vem_ops

class HypergraphCoClustering(BaseEstimator, ClusterMixin):
    """
    Variational Expectation-Maximization (VEM) algorithm for Co-clustering and
    parameter estimation on Hypergraphs.

    This class performs co-clustering of a binary hypergraph (binary M x N matrix)
    by simultaneously clustering the N nodes (columns) and M hyperedges (rows).
    It estimates the connection probabilities between clusters, and the 
    mixing proportions of each cluster for both node and hyperedge groups.

    The algorithm alternates between a Variational E-step (computing posterior
    probabilities `node_responsibilities_` and `hyperedge_responsibilities_`) 
    and an M-step (updating model parameters `node_weights_`, `hyperedge_weights_`, 
    and `block_probs_`). Model fitting and inference are handled by VEM.
    
    Model selection across different (`n_node_clusters`, `n_hyperedge_clusters`) 
    configurations is handled by sklearn.model_selection.GridSearchCV.

    Parameters
    ----------
    n_node_clusters : int, optional (default=2)
        The number of latent node clusters (K).

    n_hyperedge_clusters : int, optional (default=2)
        The number of latent hyperedge clusters (G).

    max_iter : int, optional (default=100)
        The maximum number of VEM iterations to perform during fitting.

    tol : float, optional (default=1e-4)
        The convergence tolerance. 

    n_init : int, optional (default=5)
        The number of times to restart the EM algorithm from different initializations.

    init_method : str, optional (default='kmeans')
        The initialization strategy. Options: 'kmeans', 'random', or 'choose'.
    
    init_params : dict, optional (default=None)
        Dictionary of specific parameter initialization values. Required if `init_method='choose'`,
        ignored otherwise.

    verbose : bool
        If True, prints progress information during fitting.

    prior_method : str, optional (default='weakly_informative')
        The method used to set Dirichlet and Beta priors. Options:
        - 'weakly_informative': Auto-scales Beta priors based on graph density.
        - 'flat': Sets all priors to 1.0.
        - 'jeffreys': Sets all priors to 0.5.
        - 'choose': Requires manual specification via `prior_params`.

    prior_params : dict, optional (default=None)
        Dictionary of specific prior values. Required if `prior_method='choose'`,
        ignored otherwise.

    debug : bool
        If True, enables additional diagnostic output.
    
    convergence_criterion : str
        User specified convergence criterion. 
        Current options are 'elbo' and 'parameters'. 

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    node_weights_ : np.ndarray
        Mixing proportions (priors) for the node clusters. 

    hyperedge_weights_ : np.ndarray
        Mixing proportions (priors) for the hyperedge clusters.

    block_probs_ : np.ndarray
        The connectivity matrix (Bernoulli parameters).

    node_responsibilities_ : np.ndarray
        Shape (N, `n_node_clusters`). Variational parameters (responsibilities) for nodes.

    hyperedge_responsibilities_ : np.ndarray
        Shape (M, `n_hyperedge_clusters`). Variational parameters (responsibilities) for hyperedges.

    node_clusters_ : np.ndarray
        The final hard cluster assignments for the nodes. 
        Calculated as argmax(node_responsibilities_).

    hyperedge_clusters_ : np.ndarray
        The final hard cluster assignments for the hyperedges. 
        Calculated as argmax(hyperedge_responsibilities_).

    weights_ : np.ndarray
        Alias for `node_weights_` to maintain scikit-learn compatibility.

    elbo_history_ : list
        A list of ELBO values computed during the most recent fit.
    """
    _eps = 1e-10
    _smooth = 1e-5

    def __init__(
        self,
        n_node_clusters: int = 2,
        n_hyperedge_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-8,
        n_init: int = 5,
        init_method: str = "kmeans",
        init_params: dict | None = None, 
        verbose: bool = False,
        debug: bool = False,
        prior_method: str = "weakly_informative",
        prior_params: dict | None = None,
        convergence_criterion: str = "parameters",
        random_state: int | None = None
    ):
        self.n_node_clusters = n_node_clusters
        self.n_hyperedge_clusters = n_hyperedge_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_method = init_method
        self.init_params = init_params
        self.verbose = verbose
        self.debug = debug
        self.prior_method = prior_method
        self.prior_params = prior_params
        self.convergence_criterion = convergence_criterion
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits the VEM algorithm to the hypergraph data.

        This method initializes model parameters, handles missing data (if any),
        and runs the Variational EM algorithm multiple times (`n_init`) to 
        find the best model configuration that maximizes the ICL.

        Parameters
        ----------
        X : Hypergraph or array-like
            The input data. Can be a `Hypergraph` object or an incidence matrix 
            of shape (M, N), where M is the number of hyperedges and N is the 
            number of nodes. Supports sparse matrices.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.

        Attributes Created
        ------------------
        hypergraph_ : Hypergraph
            The hypergraph object wrapping the input data.
        X_ : np.ndarray or scipy.sparse.csr_matrix
            The internal binary incidence matrix used for computation. 
            NaNs are replaced by 0 if missing data is detected.
        M_ : int
            Number of hyperedges (rows).
        N_ : int
            Number of nodes (columns).
        mask_ : np.ndarray or None
            A binary mask indicating observed values (1) vs missing values (0), 
            or None if no missing data is present.
        """
        if self.n_node_clusters < 1 or self.n_hyperedge_clusters < 1:
            raise ValueError(
                "n_node_clusters (K) and n_hyperedge_clusters (G) must be an int larger than 0."
            )

        # Handle Hypergraph Input 
        if hasattr(X, "incidence_matrix"):
            self.hypergraph_ = X
            X_to_validate = X.incidence_matrix
            self.X_ = check_array(X_to_validate, accept_sparse=['csr'], ensure_all_finite=False)
        else:
            self.X_ = check_array(X, accept_sparse=['csr'], ensure_all_finite=False)
            if self.debug:
                warnings.warn(
                    "Passed raw matrix instead of Hypergraph object. Wrapping automatically."
                )
            self.hypergraph_ = Hypergraph(self.X_)
        
        self.M_ = self.hypergraph_.num_hyperedges
        self.N_ = self.hypergraph_.num_nodes
        self.node_clusters_ = None
        self.hyperedge_clusters_ = None
        
        self.elbo_history_ = []

        self._setup_data_handling()
        self._set_priors()

        rng = check_random_state(self.random_state)

        best_run_icl = -np.inf 
        best_run_state = None # Best model state

        for run in range(self.n_init):
            state = self._run_single_em(rng)
            icl = state['icl']

            if self.verbose:
                print(f"Run {run + 1}/{self.n_init} [K={self.n_node_clusters}, G={self.n_hyperedge_clusters}]: ICL={icl:.2f}, Iter={state['iterations']}")

            if icl > best_run_icl:
                best_run_icl = icl
                best_run_state = state

        if best_run_state is not None:
            self._restore_state(best_run_state)
            self._compute_clusters()

            # Assign the row labels (hyperedge clusters) to self.labels_
            # for sklearn clustering compatibility
            self.labels_ = self.hyperedge_clusters_
            
            if self.verbose:
                print(f"Best run restored (ICL={best_run_icl:.2f}).")
        else:
            warnings.warn("All EM runs failed (NaN ICL or divergence).")

        return self

    def score(self, X: np.ndarray, y=None):
        """
        Returns the ICL score of the model.

        Parameters
        ----------
        X : np.ndarray
            The input data. Incidence matrix of shape (M, N).
            Supports sparse matrices.

        y : Ignored
            Not used, present here for API consistency by convention.
        """
        check_is_fitted(self)
        return self.compute_ICL(approximation=False)

    def _setup_data_handling(self):
        """
        Internal helper to prepare masks for missing data handling.
        
        If the incidence matrix is sparse, we assume no missing data.
        If missing data is present, creates a mask where 1=observed, 0=missing.
        It also displays a warning message to alert the user if self.debug = True.
        """
        # If Sparse, we assume no missing data
        if sparse.issparse(self.X_):
            self.has_missing_data_ = False
            self.mask_ = None
            if self.debug:
                print("Sparse input detected.")
            return
        
        # If dense, check for NaN values
        self.has_missing_data_ = np.isnan(self.X_).any()

        # Save a mask which shows which entries were not NaN originally
        # This is used to distinguish NaN 0s from observed 0s
        if self.has_missing_data_:
            self.mask_ = (~np.isnan(self.X_)).astype(float)
            self.X_ = np.nan_to_num(self.X_, copy=True, nan=0)
            if self.debug:
                print("Missing data detected.")       
        else:
            self.mask_ = None
            if self.debug: 
                print("No missing data.")

    def _set_priors(self):
        """
        Sets the prior hyperparameters for the VEM model by calling `vem_ops.get_prior_parameters`.

        Updates the internal prior attributes (`self.prior_dirichlet_alpha_`,
        `self.prior_beta_alpha_`, `self.prior_beta_beta_`).
        """
        da, ba, bb = vem_ops.get_prior_parameters(
            X=self.X_,
            method=self.prior_method,
            mask=self.mask_,
            prior_params=self.prior_params,
        )

        self.prior_dirichlet_alpha_ = da
        self.prior_beta_alpha_ = ba
        self.prior_beta_beta_ = bb

    def _initialize_parameters(self, rng):
        """
        Initializes node/hyperedge responsibilities and model parameters.

        Parameters
        ----------
        rng : numpy.random.RandomState
            The random number generator instance used for this specific run.
        """
        priors_tuple = (
            self.prior_dirichlet_alpha_,
            self.prior_beta_alpha_,
            self.prior_beta_beta_
        )

        # vem_ops returns standard tuple, we map to descriptive names
        c_init, d_init, gamma_init, delta_init, theta_init = (
            vem_ops.initialize_parameters(
                self.X_,
                self.n_node_clusters,
                self.n_hyperedge_clusters,
                self.init_method,
                smooth=self._smooth,
                priors=priors_tuple,
                params=self.init_params,
                random_state = rng
            )
        )

        self.node_responsibilities_ = c_init
        self.hyperedge_responsibilities_ = d_init
        self.node_weights_ = gamma_init 
        self.hyperedge_weights_ = delta_init 
        self.block_probs_ = theta_init 

        # KMeans/Kmodes only initializes c_ and d_, so we run an M-step to also initialize
        # node_weights_ (gamma), hyperedge_weights_ (delta), block_probs_ (theta).
        if self.node_weights_ is None:
            self._m_step()

    def _ve_step(self):
        """
        Wrapper for vem_ops.compute_responsibilities.
        Performs the Variational E-step to update responsibilities
        based on current parameters.
        """
        self.node_responsibilities_ = vem_ops.compute_responsibilities(
            X=self.X_,
            responsibilities=self.hyperedge_responsibilities_,
            theta=self.block_probs_,
            mixing_proportions=self.node_weights_,
            mask=self.mask_,
            mode="node",
            eps=self._eps,
        )
        self.hyperedge_responsibilities_ = vem_ops.compute_responsibilities(
            X=self.X_,
            responsibilities=self.node_responsibilities_,
            theta=self.block_probs_,
            mixing_proportions=self.hyperedge_weights_,
            mask=self.mask_,
            mode="hyperedge",
            eps=self._eps,
        )

    def _m_step(self):
        """
        Wrapper for vem_ops.M_step. 
        Performs the M-step to update model parameters based on current responsibilities.
        """
        priors = (
            self.prior_dirichlet_alpha_,
            self.prior_beta_alpha_,
            self.prior_beta_beta_,
        )

        self.node_weights_, self.hyperedge_weights_, self.block_probs_ = vem_ops.M_step(
            X=self.X_,
            c=self.node_responsibilities_,
            d=self.hyperedge_responsibilities_,
            mask=self.mask_,
            priors=priors,
            K=self.n_node_clusters,
            G=self.n_hyperedge_clusters,
            eps=self._eps,
        )

    def _run_single_em(self, rng):
        """
        Executes a single run of the Variational EM algorithm until convergence.

        This method initializes parameters, then alternates between the 
        Variational E-step (computing posterior probabilities) and the M-step 
        (updating model parameters) until the change in parameters is below 
        `tol` or `max_iter` is reached.

        Parameters
        ----------
        rng : numpy.random.RandomState
            The random number generator instance used for this specific run.

        Returns
        -------
        state : dict
            A dictionary containing the final model state for this run.
        """
        self.elbo_history_ = []
        self._initialize_parameters(rng)

        converged = False
        icl = -np.inf

        for iteration in range(self.max_iter):
            old_params = (self.block_probs_.copy(), self.node_weights_.copy(), self.hyperedge_weights_.copy())

            self._ve_step()
            self._m_step()

            if self.debug or self.convergence_criterion == "elbo":
                self.compute_ELBO()
            
            # Current ELBO is always the last one added
            elbo_curr = self.elbo_history_[-1] if self.elbo_history_ else None
    
            # Old ELBO is the second to last, but only if we have at least 2 entries
            elbo_prev = self.elbo_history_[-2] if len(self.elbo_history_) > 1 else -np.inf

            # Convergence is met either by small changes in parameters/elbo, or by 
            # reaching the maximum number of iterations, max_iter.
            if iteration > 0:
                converged = vem_ops.check_convergence(
                    params_current=(self.block_probs_, self.node_weights_, self.hyperedge_weights_),
                    params_old=old_params,
                    tol=self.tol,
                    iteration=iteration + 1,
                    max_iter=self.max_iter,
                    criterion=self.convergence_criterion,
                    elbo_old=elbo_prev,
                    elbo_current=elbo_curr
                )

            if converged:
                icl = self.compute_ICL(approximation=False)
                if not self.debug:
                    self.compute_ELBO()  # Ensure final ELBO is recorded
                break

        state = {
            "block_probs": self.block_probs_.copy(),
            "node_weights": self.node_weights_.copy(),
            "hyperedge_weights": self.hyperedge_weights_.copy(),
            "K": self.n_node_clusters,
            "G": self.n_hyperedge_clusters,
            "node_resp": self.node_responsibilities_.copy(),
            "hyperedge_resp": self.hyperedge_responsibilities_.copy(),
            "icl": icl,
            "elbo_history": self.elbo_history_.copy(),
            "iterations": iteration + 1,
        }
        return state

    def _restore_state(self, state: dict):
        """
        Restores model attributes from a dictionary state.
        """
        self.block_probs_ = state["block_probs"]
        self.node_weights_ = state["node_weights"]
        self.hyperedge_weights_ = state["hyperedge_weights"]
        self.node_responsibilities_ = state["node_resp"]
        self.hyperedge_responsibilities_ = state["hyperedge_resp"]
        self.elbo_history_ = state["elbo_history"]

    def compute_ELBO(self) -> float:
        """
        Computes the Evidence Lower Bound (ELBO) and updates the history.
        
        This method calculates the ELBO based on current variational parameters
        and appends the result to `self.elbo_history_`.
        """
        check_is_fitted(self)
        elbo = icl_ops.compute_elbo(
            self.X_, self.mask_, self.node_weights_, self.hyperedge_weights_, 
            self.block_probs_, self.node_responsibilities_, self.hyperedge_responsibilities_
        )
        self.elbo_history_.append(elbo)
        return elbo

    def compute_ICL(self, approximation: bool = False) -> float:
        """
        Computes the Integrated Complete-data Likelihood (ICL).

        Parameters
        ----------
        approximation : bool, optional (default=False)
            If True, computes a BIC-like approximation using the penalized ELBO.
            If False, computes the exact ICL value using hard cluster assignments.

        Returns
        -------
        float
            The ICL score.
        """
        check_is_fitted(self)
        return icl_ops.compute_icl(
            X=self.X_,
            mask=self.mask_,
            c=self.node_responsibilities_,
            d=self.hyperedge_responsibilities_,
            K=self.n_node_clusters,
            G=self.n_hyperedge_clusters,
            alpha_dirichlet=self.prior_dirichlet_alpha_,
            alpha_beta=self.prior_beta_alpha_,
            beta_beta=self.prior_beta_beta_,
            gamma=self.node_weights_,
            delta=self.hyperedge_weights_,
            theta=self.block_probs_,
            approximation=approximation,
        )

    def _compute_clusters(self):
        """
        Computes hard cluster assignments based on responsibilities.
        """
        self.node_clusters_ = np.argmax(self.node_responsibilities_, axis=1)
        self.hyperedge_clusters_ = np.argmax(self.hyperedge_responsibilities_, axis=1)

    def predict_proba(self, X):
        """
        Predicts soft cluster probabilities for (new) hyperedges (rows) in X.
        
        If X is new data, computes the posterior `hyperedge_responsibilities` based on 
        fixed model parameters (a projection or partial VE-step).

        Requires X to have N_ columns (i.e. include all the nodes present in training)

        Parameters
        ----------
        X : np.ndarray or scipy.sparse.csr_matrix
            A hypergraph including N_ nodes and any number of (new) hyperedges.
            
        Returns
        -------
        np.ndarray
            An array with the responsibilities of X.
        """
        check_is_fitted(self)

        if X.shape[1] != self.N_:
            raise ValueError(f"Input has {X.shape[1]} nodes, but model was trained on {self.N_} nodes.")

        # Handle input types (Hypergraph vs Matrix)
        if hasattr(X, "incidence_matrix"):
            X_mat = X.incidence_matrix
        else:
            X_mat = X
        
        # We need a mask for the new X if it has NaNs
        if sparse.issparse(X_mat):
            mask = None
        else:
            mask = (~np.isnan(X_mat)).astype(float) if np.isnan(X_mat).any() else None
            X_mat = np.nan_to_num(X_mat, nan=0)

        # Projection (Partial VE-step)
        # We calculate 'hyperedge_responsibilities' using the fixed parameters
        # from training.
        probas = vem_ops.compute_responsibilities(
            X=X_mat,
            responsibilities=self.node_responsibilities_,  
            theta=self.block_probs_,       
            mixing_proportions=self.hyperedge_weights_, 
            mask=mask,
            mode="hyperedge",
            eps=self._eps
        )
        
        return probas
    
    # Aliases

    @property
    def weights_(self):
        """
        Alias for `node_weights_` to maintain scikit-learn compatibility.
        Downstream tools expecting a mixture model will find the row (hyperedge) weights here.
        """
        return self.hyperedge_weights_

    @property
    def gamma_(self):
        """Alias for `node_weights_` (Math notation)."""
        return self.node_weights_
    
    @property
    def delta_(self):
        """Alias for `hyperedge_weights_` (Math notation)."""
        return self.hyperedge_weights_
    
    @property
    def theta_(self):
        """Alias for `block_probs_` (Math notation)."""
        return self.block_probs_
    
    @property
    def c_(self):
        """Alias for `node_responsibilities_` (Math notation)."""
        return self.node_responsibilities_

    @property
    def d_(self):
        """Alias for `hyperedge_responsibilities_` (Math notation)."""
        return self.hyperedge_responsibilities_
