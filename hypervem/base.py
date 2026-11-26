"""
Base estimator class for Hypergraph Co-clustering models in hypervem.
"""

from __future__ import annotations
import abc
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted


class BaseCoClustering(BaseEstimator, ClusterMixin, abc.ABC):
    """
    Abstract base class for Variational EM Hypergraph Co-clustering estimators.

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
        The initialization strategy.

    verbose : bool, optional (default=False)
        If True, prints progress information during fitting.

    random_state : int, RandomState instance or None, optional (default=None)
        Seed or random state for reproducibility.
    """

    def __init__(
        self,
        n_node_clusters: int = 2,
        n_hyperedge_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        init_method: str = "kmeans",
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.n_node_clusters = n_node_clusters
        self.n_hyperedge_clusters = n_hyperedge_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_method = init_method
        self.verbose = verbose
        self.random_state = random_state

    def _validate_cluster_numbers(self):
        if self.n_node_clusters < 1 or self.n_hyperedge_clusters < 1:
            raise ValueError(
                "n_node_clusters (K) and n_hyperedge_clusters (G) must be >= 1."
            )

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit the hypergraph co-clustering model to data."""

    @property
    def labels_(self) -> np.ndarray:
        """Alias for hyperedge cluster assignments to maintain sklearn ClusterMixin compatibility."""
        check_is_fitted(self, "hyperedge_clusters_")
        return self.hyperedge_clusters_

    @property
    def hyperedge_labels_(self) -> np.ndarray:
        """Alias for hyperedge cluster assignments."""
        check_is_fitted(self, "hyperedge_clusters_")
        return self.hyperedge_clusters_

    @property
    def node_labels_(self) -> np.ndarray:
        """Alias for node cluster assignments."""
        check_is_fitted(self, "node_clusters_")
        return self.node_clusters_

    @property
    def coclusters_(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Co-cluster membership indicators: `(hyperedges, nodes)` boolean masks of
        shape `(n_hyperedge_clusters, n_hyperedges)` and `(n_node_clusters, n_nodes)`.
        """
        check_is_fitted(self, ["hyperedge_clusters_", "node_clusters_"])
        hyperedges = self.hyperedge_clusters_[None, :] == np.arange(self.n_hyperedge_clusters)[:, None]
        nodes = self.node_clusters_[None, :] == np.arange(self.n_node_clusters)[:, None]
        return hyperedges, nodes

    def _compute_clusters(self):
        """Computes hard node and hyperedge cluster assignments from responsibilities."""
        self.node_clusters_ = np.argmax(self.node_responsibilities_, axis=1)
        self.hyperedge_clusters_ = np.argmax(self.hyperedge_responsibilities_, axis=1)

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fits the model and returns the hyperedge cluster assignments."""
        return self.fit(X).labels_

    def get_cocluster(self, g: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the hyperedge and node indices belonging to co-cluster `(g, k)`."""
        check_is_fitted(self, ["hyperedge_clusters_", "node_clusters_"])
        hyperedge_indices = np.where(self.hyperedge_clusters_ == g)[0]
        node_indices = np.where(self.node_clusters_ == k)[0]
        return hyperedge_indices, node_indices
