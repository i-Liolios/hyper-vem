import numpy as np
import pytest
from scipy import sparse
from hypervem.model import HypergraphCoClustering
from hypervem.hypergraph import Hypergraph

@pytest.fixture
def binary_matrix():
    np.random.seed(10)
    return np.random.randint(0, 2, size=(20, 25))

def test_estimator_init():
    vem = HypergraphCoClustering(n_node_clusters=3, n_hyperedge_clusters=4)
    assert vem.n_node_clusters == 3
    assert vem.n_hyperedge_clusters == 4
    assert vem.init_method == "kmeans"

def test_fit_attributes(binary_matrix):
    """Test that fit populates all required attributes."""
    vem = HypergraphCoClustering(
        n_node_clusters=2, 
        n_hyperedge_clusters=2, 
        max_iter=5, 
        random_state=42,
        verbose=False
    )
    
    vem.fit(binary_matrix)
    
    # Check presence of attributes
    assert hasattr(vem, "node_responsibilities_")
    assert hasattr(vem, "hyperedge_responsibilities_")
    assert hasattr(vem, "block_probs_")
    assert hasattr(vem, "node_clusters_")
    assert hasattr(vem, "hyperedge_clusters_")
    
    # Check shapes
    assert vem.node_responsibilities_.shape == (25, 2)
    assert vem.hyperedge_responsibilities_.shape == (20, 2)
    assert vem.block_probs_.shape == (2, 2)
    
    # Check Labels
    assert len(vem.hyperedge_clusters_) == 20 
    assert len(vem.node_clusters_) == 25

def test_fit_sparse_input(binary_matrix):
    """Test that the model accepts sparse matrices."""
    X_sparse = sparse.csr_matrix(binary_matrix)
    vem = HypergraphCoClustering(n_node_clusters=2, n_hyperedge_clusters=2, max_iter=2)
    
    # Should run without error
    vem.fit(X_sparse)
    
    assert vem.M_ == 20
    assert vem.N_ == 25

def test_predict_proba(binary_matrix):
    """Test inference on new hyperedges."""
    vem = HypergraphCoClustering(n_node_clusters=2, n_hyperedge_clusters=2, max_iter=5, random_state=42)
    vem.fit(binary_matrix)
    
    # Create "new" hyperedges (must have same number of nodes/columns)
    new_edges = np.random.randint(0, 2, size=(5, 25))
    
    probas = vem.predict_proba(new_edges)
    
    assert probas.shape == (5, 2) # (n_samples, n_hyperedge_clusters)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)

def test_score_returns_float(binary_matrix):
    """Test that scoring (ICL) returns a number."""
    vem = HypergraphCoClustering(n_node_clusters=2, n_hyperedge_clusters=2, max_iter=2)
    vem.fit(binary_matrix)
    score = vem.score(binary_matrix)
    assert isinstance(score, float)
    assert not np.isnan(score)

def test_elbo_increases(binary_matrix):
    """Test that ELBO generally increases or converges."""
    vem = HypergraphCoClustering(
        n_node_clusters=2, 
        n_hyperedge_clusters=2, 
        max_iter=20, 
        random_state=42,
        tol=1e-10,
        convergence_criterion='elbo'
    )
    vem.fit(binary_matrix)
    
    history = vem.elbo_history_
    assert len(history) > 1
    # Check if the last ELBO is better than the first
    assert history[-1] >= history[0]