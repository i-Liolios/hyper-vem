import numpy as np
from scipy import sparse

class Hypergraph:
    """
    Represents a hypergraph for co-clustering models.

    A hypergraph consists of nodes and hyperedges, with an incidence matrix
    indicating which nodes belong to which hyperedges. This class supports
    both real and synthetically generated hypergraphs.

    Attributes
    ----------
    num_hyperedges : int
        Number of hyperedges (rows of incidence matrix).

    num_nodes : int
        Number of nodes (columns of incidence matrix).

    incidence_matrix : np.ndarray or scipy.sparse.csr_array
        Incidence matrix of shape (num_hyperedges, num_nodes) with binary entries.

    ground_truth : dict, optional
        Ground truth parameters for synthetic data including cluster assignments 
        and connectivity probabilities.
    """

    def __init__(self, incidence_matrix: np.ndarray | sparse.csr_array, ground_truth: dict | None = None):
        """
        Initialization method for Hypergraph class.

        Parameters
        ----------
        incidence_matrix : np.ndarray or scipy.sparse.csr_array
            Incidence matrix of shape (num_hyperedges x num_nodes).

        ground_truth : dict, optional
            Ground truth parameters if available.
        """
        self.incidence_matrix = incidence_matrix
        self.num_nodes = self.incidence_matrix.shape[1] 
        self.num_hyperedges = self.incidence_matrix.shape[0]  
        self.ground_truth = ground_truth

    @classmethod
    def generate_random(
        cls,
        num_hyperedges: int,
        num_nodes: int,
        total_node_groups: int = 2,
        total_hyperedge_groups: int = 2,
        gamma: np.ndarray | None = None,
        delta: np.ndarray | None = None,
        theta: np.ndarray | None = None,
    ):
        """
        Creates a synthetic hypergraph instance using a Hypergraph Co-clustering Model.

        Generates data where each node is assigned to one of `total_node_groups` clusters and each
        hyperedge to one of `total_hyperede_groups` groups. Connections are sampled from a Bernoulli
        distribution based on block interaction probabilities (`theta`).

        Parameters
        ----------
        num_hyperedges : int
            Number of hyperedges (rows).

        num_nodes : int
            Number of nodes (columns).

        total_node_groups : int, optional (default=2)
            Number of node clusters.

        total_hyperedge_groups : int, optional (default=2)
            Number of hyperedge groups.

        gamma : np.ndarray, optional
            Node cluster proportions (K,). Defaults to uniform Dirichlet.

        delta : np.ndarray, optional
            Hyperedge group proportions (G,). Defaults to uniform Dirichlet.

        theta : np.ndarray, optional
            Block interaction probabilities (K, G). Defaults to Beta(1,1) random samples.

        Returns
        -------
        Hypergraph
            An instance containing the synthetic incidence matrix `X` and a `ground_truth` dictionary.
        """
        # Generate data through the data generating process
        if gamma is None:
            gamma = np.random.dirichlet(np.ones(total_node_groups))
        if delta is None:
            delta = np.random.dirichlet(np.ones(total_hyperedge_groups))
        if theta is None:
            theta = np.random.beta(1, 1, size=(total_node_groups, total_hyperedge_groups)) 

        z_hyperedges = np.random.choice(total_hyperedge_groups, size=num_hyperedges, p=delta)
        z_nodes = np.random.choice(total_node_groups, size=num_nodes, p=gamma)

        # X[j, i] ~ Bernoulli(theta[group_of_j, group_of_i])

        idx_nodes = [np.flatnonzero(z_nodes == k) for k in range(total_node_groups)]
        idx_hyperedges = [np.flatnonzero(z_hyperedges == g) for g in range(total_hyperedge_groups)]

        incidence_matrix = np.zeros((num_hyperedges, num_nodes), dtype=np.int8)

        for k, cols in enumerate(idx_nodes):
            for g, rows in enumerate(idx_hyperedges):
                incidence_matrix[np.ix_(rows, cols)] = np.random.binomial(
                    1, theta[k, g], size=(rows.size, cols.size)
                )

        ground_truth = {
            "gamma": gamma,
            "delta": delta,
            "theta": theta,
            "z_hyperedges": z_hyperedges,
            "z_nodes": z_nodes,
        }

        return cls(incidence_matrix=incidence_matrix, ground_truth=ground_truth)
    
