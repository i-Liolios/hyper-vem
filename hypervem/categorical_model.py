import warnings
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import BaseCoClustering
from .hypergraph import Hypergraph
from . import categorical_vem_ops as cat_ops
from . import vem_ops


class CategoricalCoClustering(BaseCoClustering):
    """
    Variational Expectation-Maximization (VEM) algorithm for Co-clustering and
    parameter estimation on Categorical Hypergraphs.

    This class performs co-clustering of a categorical hypergraph (M x N
    incidence matrix) by simultaneously clustering the N nodes (columns) and
    M hyperedges (rows) where each observed incidence takes one of C discrete
    categories. Missing entries (np.nan) are treated as unobserved edges.

    The algorithm alternates between a Variational E-step (computing posterior
    probabilities `node_responsibilities_` and `hyperedge_responsibilities_`)
    and an M-step (updating model parameters `node_weights_`, `hyperedge_weights_`,
    and `block_probs_`).

    Model selection across different (`n_node_clusters`, `n_hyperedge_clusters`)
    configurations is handled by sklearn.model_selection.GridSearchCV.

    Parameters
    ----------
    n_node_clusters : int, optional (default=2)
        The number of latent node clusters (K).

    n_hyperedge_clusters : int, optional (default=2)
        The number of latent hyperedge clusters (G).

    n_categories : int, optional (default=None)
        Number of distinct edge categories (C). If None, inferred as max(X) + 1.

    max_iter : int, optional (default=100)
        The maximum number of VEM iterations to perform during fitting.

    tol : float, optional (default=1e-4)
        The convergence tolerance.

    n_init : int, optional (default=5)
        The number of times to restart the EM algorithm from different initializations.

    init_method : str, optional (default='kmeans')
        The initialization strategy. Options: 'kmeans', 'random', or 'choose'.

    init_params : dict, optional (default=None)
        Dictionary of initial parameter values when `init_method='choose'`.

    verbose : bool, optional (default=False)
        If True, prints progress information during fitting.

    debug : bool, optional (default=False)
        If True, enables additional diagnostic output.

    alpha_dirichlet : float, optional (default=1.0)
        Symmetric Dirichlet prior hyperparameter (1.0 = flat prior / MLE).

    convergence_criterion : str, optional (default='parameters')
        Stopping criterion for VEM. Current options are 'parameters' and 'elbo'.

    random_state : int, RandomState instance or None, optional (default=None)
        Seed or random state for reproducibility.

    Attributes
    ----------
    node_weights_ : np.ndarray
        Mixing proportions (priors) for the node clusters (shape K,).

    hyperedge_weights_ : np.ndarray
        Mixing proportions (priors) for the hyperedge clusters (shape G,).

    block_probs_ : np.ndarray
        The categorical connectivity tensor pi (shape K, G, C).

    node_responsibilities_ : np.ndarray
        Shape (N, `n_node_clusters`). Variational parameters (responsibilities) for nodes.

    hyperedge_responsibilities_ : np.ndarray
        Shape (M, `n_hyperedge_clusters`). Variational parameters (responsibilities) for hyperedges.

    node_clusters_ : np.ndarray
        Hard cluster assignments for the nodes, calculated as argmax(node_responsibilities_).

    hyperedge_clusters_ : np.ndarray
        Hard cluster assignments for the hyperedges, calculated as argmax(hyperedge_responsibilities_).

    weights_ : np.ndarray
        Alias for `hyperedge_weights_` to maintain scikit-learn compatibility.

    elbo_ : float
        Final Evidence Lower Bound of the winning restart.

    icl_ : float
        Exact Integrated Complete-data Likelihood of the winning model.

    elbo_history_ : list
        A list of ELBO values computed during the most recent fit.
    """

    _eps = 1e-12
    _smooth = 1e-5

    def __init__(
        self,
        n_node_clusters: int = 2,
        n_hyperedge_clusters: int = 2,
        n_categories: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        init_method: str = "kmeans",
        init_params: dict | None = None,
        verbose: bool = False,
        debug: bool = False,
        alpha_dirichlet: float = 1.0,
        convergence_criterion: str = "parameters",
        random_state: int | None = None,
    ):
        super().__init__(
            n_node_clusters=n_node_clusters,
            n_hyperedge_clusters=n_hyperedge_clusters,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            init_method=init_method,
            verbose=verbose,
            random_state=random_state,
        )
        self.n_categories = n_categories
        self.init_params = init_params
        self.debug = debug
        self.alpha_dirichlet = alpha_dirichlet
        self.convergence_criterion = convergence_criterion

    def fit(self, X, y=None):
        """
        Fits the Categorical VEM algorithm to the hypergraph data.

        Parameters
        ----------
        X : Hypergraph, array-like, or scipy.sparse matrix of shape (M, N)
            The input hypergraph or discrete incidence matrix with entries in
            {0, ..., C-1} or np.nan.

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

        X_cats_ : list of scipy.sparse.csr_array
            Per-category binary indicator matrices of shape (M, N).

        mask_ : np.ndarray
            Boolean observation mask indicating non-missing entries.

        M_ : int
            Number of hyperedges (rows).

        N_ : int
            Number of nodes (columns).
        """
        self._validate_cluster_numbers()
        if self.convergence_criterion not in ("parameters", "elbo"):
            raise ValueError(f"Unknown convergence_criterion: {self.convergence_criterion}")

        self._setup_data_handling(X)
        rng = check_random_state(self.random_state)

        best_run_elbo = -np.inf
        best_run_state = None

        for run in range(self.n_init):
            state = self._run_single_em(rng)
            elbo = state["elbo"]

            if self.verbose:
                print(
                    f"Run {run + 1}/{self.n_init} [K={self.n_node_clusters}, "
                    f"G={self.n_hyperedge_clusters}, C={self.n_categories_}]: "
                    f"ELBO={elbo:.2f}, ICL={state['icl']:.2f}, Iter={state['iterations']}"
                )

            if np.isfinite(elbo) and elbo > best_run_elbo:
                best_run_elbo = elbo
                best_run_state = state

        if best_run_state is not None:
            self._restore_state(best_run_state)
            self._compute_clusters()
            self.elbo_ = best_run_elbo
            self.icl_ = best_run_state["icl"]
            self.n_iter_ = best_run_state["iterations"]

            if self.verbose:
                print(f"Best run restored (ELBO={self.elbo_:.2f}, ICL={self.icl_:.2f}).")
        else:
            warnings.warn("All EM runs failed (NaN ELBO or divergence).")

        return self

    def score(self, X, y=None):
        """
        Returns the ICL score of the model.

        Parameters
        ----------
        X : Hypergraph, array-like, or scipy.sparse matrix
            Incidence matrix of shape (M, N).

        y : Ignored
            Not used, present here for API consistency by convention.
        """
        check_is_fitted(self, "icl_")
        return self.icl_

    def _setup_data_handling(self, X):
        """
        Internal helper to wrap the hypergraph, build per-category indicators,
        and detect missing entries.
        """
        if hasattr(X, "incidence_matrix"):
            self.hypergraph_ = X
        else:
            self.hypergraph_ = Hypergraph(X)

        X_cats, mask, categories, C = cat_ops.prepare_categorical_matrices(
            self.hypergraph_, n_categories=self.n_categories
        )
        self.X_cats_ = X_cats
        self.mask_ = mask
        self.categories_ = categories
        self.n_categories_ = C
        self.M_, self.N_ = self.hypergraph_.num_hyperedges, self.hypergraph_.num_nodes
        self.has_missing_data_ = bool(not np.all(mask))

        if self.debug:
            print("Missing data detected." if self.has_missing_data_ else "No missing data.")

    def _initialize_parameters(self, rng):
        """
        Initializes node/hyperedge responsibilities and categorical model parameters.
        """
        c_init, d_init, gamma_init, delta_init, pi_init = cat_ops.initialize_categorical_parameters(
            X_cats=self.X_cats_,
            mask=self.mask_,
            K=self.n_node_clusters,
            G=self.n_hyperedge_clusters,
            C=self.n_categories_,
            method=self.init_method,
            smooth=self._smooth,
            params=self.init_params,
            random_state=rng,
        )

        self.node_responsibilities_ = c_init
        self.hyperedge_responsibilities_ = d_init
        self.node_weights_ = gamma_init
        self.hyperedge_weights_ = delta_init
        self.block_probs_ = pi_init
        self._last_N_kgc_ = None

        if self.node_weights_ is None:
            self._m_step()

    def _ve_step(self):
        """
        Performs the Variational E-step to update responsibilities based on current parameters.
        """
        self.node_responsibilities_ = cat_ops.compute_categorical_responsibilities(
            X_cats=self.X_cats_,
            responsibilities=self.hyperedge_responsibilities_,
            pi=self.block_probs_,
            mixing_proportions=self.node_weights_,
            mode="node",
            eps=self._eps,
        )
        self.hyperedge_responsibilities_ = cat_ops.compute_categorical_responsibilities(
            X_cats=self.X_cats_,
            responsibilities=self.node_responsibilities_,
            pi=self.block_probs_,
            mixing_proportions=self.hyperedge_weights_,
            mode="hyperedge",
            eps=self._eps,
        )

    def _m_step(self):
        """
        Performs the M-step to update model parameters based on current responsibilities.
        """
        self._last_N_kgc_ = cat_ops._block_counts(
            self.X_cats_, self.node_responsibilities_, self.hyperedge_responsibilities_
        )
        self.node_weights_, self.hyperedge_weights_, self.block_probs_ = cat_ops.categorical_m_step(
            X_cats=self.X_cats_,
            c=self.node_responsibilities_,
            d=self.hyperedge_responsibilities_,
            K=self.n_node_clusters,
            G=self.n_hyperedge_clusters,
            C=self.n_categories_,
            alpha_dirichlet_mix=self.alpha_dirichlet,
            alpha_dirichlet_pi=self.alpha_dirichlet,
            N_kgc=self._last_N_kgc_,
            eps=self._eps,
        )

    def _run_single_em(self, rng):
        """
        Executes a single run of the Variational EM algorithm until convergence.
        """
        self.elbo_history_ = []
        self._initialize_parameters(rng)

        converged = False
        icl = -np.inf
        iteration = 0

        for iteration in range(self.max_iter):
            old_params = (
                self.block_probs_.copy(),
                self.node_weights_.copy(),
                self.hyperedge_weights_.copy(),
            )

            self._ve_step()
            self._m_step()
            self.compute_ELBO()

            elbo_curr = self.elbo_history_[-1]
            elbo_prev = self.elbo_history_[-2] if len(self.elbo_history_) > 1 else -np.inf

            if iteration > 0 or self.max_iter == 1:
                converged = vem_ops.check_convergence(
                    params_current=(self.block_probs_, self.node_weights_, self.hyperedge_weights_),
                    params_old=old_params,
                    tol=self.tol,
                    iteration=iteration + 1,
                    max_iter=self.max_iter,
                    criterion=self.convergence_criterion,
                    elbo_old=elbo_prev,
                    elbo_current=elbo_curr,
                )

            if converged and (iteration >= 2 or iteration + 1 == self.max_iter):
                icl = self.compute_ICL(approximation=False)
                break

        elbo = self.elbo_history_[-1] if self.elbo_history_ else -np.inf
        return {
            "block_probs": self.block_probs_.copy(),
            "node_weights": self.node_weights_.copy(),
            "hyperedge_weights": self.hyperedge_weights_.copy(),
            "node_resp": self.node_responsibilities_.copy(),
            "hyperedge_resp": self.hyperedge_responsibilities_.copy(),
            "elbo": elbo,
            "icl": icl,
            "elbo_history": self.elbo_history_.copy(),
            "iterations": iteration + 1 if self.max_iter > 0 else 0,
        }

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
        self._last_N_kgc_ = None

    def compute_ELBO(self) -> float:
        """
        Computes the Evidence Lower Bound (ELBO) and updates `self.elbo_history_`.
        """
        check_is_fitted(self, ["X_cats_", "block_probs_"])
        elbo = cat_ops.compute_categorical_elbo(
            self.X_cats_,
            self.node_weights_,
            self.hyperedge_weights_,
            self.block_probs_,
            self.node_responsibilities_,
            self.hyperedge_responsibilities_,
            N_kgc=getattr(self, "_last_N_kgc_", None),
            eps=self._eps,
        )
        self.elbo_history_.append(elbo)
        return elbo

    def compute_ICL(self, approximation: bool = False) -> float:
        """
        Computes the Integrated Complete-data Likelihood (ICL).

        Parameters
        ----------
        approximation : bool, optional (default=False)
            If True, computes a BIC-like approximation using penalized Expected CDLL.
            If False, computes exact ICL using hard cluster assignments.

        Returns
        -------
        float
            The ICL score.
        """
        check_is_fitted(self, ["X_cats_", "block_probs_"])
        return cat_ops.compute_categorical_icl(
            X_cats=self.X_cats_,
            c=self.node_responsibilities_,
            d=self.hyperedge_responsibilities_,
            K=self.n_node_clusters,
            G=self.n_hyperedge_clusters,
            C=self.n_categories_,
            gamma=self.node_weights_,
            delta=self.hyperedge_weights_,
            pi=self.block_probs_,
            alpha_dirichlet=self.alpha_dirichlet,
            approximation=approximation,
            eps=self._eps,
        )

    def predict_proba(self, X) -> np.ndarray:
        """
        Predicts soft cluster probabilities for new hyperedges in X.

        Parameters
        ----------
        X : Hypergraph, array-like, or scipy.sparse matrix
            A hypergraph including `N_` nodes and any number of new hyperedges.

        Returns
        -------
        np.ndarray
            Posterior hyperedge cluster responsibilities of shape (M_new, G).
        """
        check_is_fitted(self, ["node_responsibilities_", "block_probs_", "hyperedge_weights_"])
        X_cats_new, _, _, _ = cat_ops.prepare_categorical_matrices(
            X, n_categories=self.n_categories_
        )
        if X_cats_new[0].shape[1] != self.N_:
            raise ValueError(
                f"Input has {X_cats_new[0].shape[1]} nodes, but model was trained on {self.N_} nodes."
            )

        return cat_ops.compute_categorical_responsibilities(
            X_cats=X_cats_new,
            responsibilities=self.node_responsibilities_,
            pi=self.block_probs_,
            mixing_proportions=self.hyperedge_weights_,
            mode="hyperedge",
            eps=self._eps,
        )

    # Aliases

    @property
    def weights_(self):
        """Alias for `hyperedge_weights_` to maintain scikit-learn compatibility."""
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
    def pi_(self):
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
