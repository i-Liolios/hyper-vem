import numpy as np
import pytest
from hypervem.vem_ops import initialize_parameters, compute_responsibilities, M_step, check_convergence

@pytest.fixture
def synthetic_data():
    """Creates a small 10x12 binary matrix for testing."""
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(10, 12))
    return X

def test_initialization_shapes(synthetic_data):
    """Test that initialization returns correct shapes."""
    M, N = synthetic_data.shape
    K, G = 2, 3
    
    c, d, gamma, delta, theta = initialize_parameters(
        synthetic_data, K, G, method="random", random_state=42
    )
    
    assert c.shape == (N, K)
    assert d.shape == (M, G)
    assert gamma.shape == (K,)
    assert delta.shape == (G,)
    assert theta.shape == (K, G)
    
    # Check normalization
    np.testing.assert_allclose(c.sum(axis=1), 1.0)
    np.testing.assert_allclose(d.sum(axis=1), 1.0)

def test_compute_responsibilities_normalization(synthetic_data):
    """Test that the E-step produces valid probability distributions."""
    M, N = synthetic_data.shape
    K, G = 2, 2
    
    # Mock inputs
    c = np.full((N, K), 1/K)
    d = np.full((M, G), 1/G)
    theta = np.array([[0.8, 0.2], [0.2, 0.8]])
    gamma = np.array([0.5, 0.5])
    delta = np.array([0.5, 0.5])
    
    # Test Node Mode
    new_c = compute_responsibilities(
        synthetic_data, d, theta, gamma, mask=None, mode="node"
    )
    assert new_c.shape == (N, K)
    np.testing.assert_allclose(new_c.sum(axis=1), 1.0)
    assert np.all(new_c >= 0) and np.all(new_c <= 1)

    # Test Hyperedge Mode
    new_d = compute_responsibilities(
        synthetic_data, c, theta, delta, mask=None, mode="hyperedge"
    )
    assert new_d.shape == (M, G)
    np.testing.assert_allclose(new_d.sum(axis=1), 1.0)

def test_m_step_bounds(synthetic_data):
    """Test that M-step parameters stay within bounds."""
    M, N = synthetic_data.shape
    K, G = 2, 2
    
    # Random responsibilities
    c = np.random.dirichlet([1]*K, size=N)
    d = np.random.dirichlet([1]*G, size=M)
    priors = (1.05, 1.0, 1.0) # Dirichlet, Beta_a, Beta_b
    
    gamma, delta, theta = M_step(synthetic_data, c, d, None, priors, K, G)
    
    # Assertions
    assert np.all(theta >= 0) and np.all(theta <= 1)
    np.testing.assert_allclose(gamma.sum(), 1.0)
    np.testing.assert_allclose(delta.sum(), 1.0)

def test_check_convergence():
    """Test the convergence logic."""
    # Identical parameters should return True
    params = (np.zeros(2), np.zeros(2), np.zeros(2))
    assert check_convergence(params, params, tol=1e-3, criterion="parameters") == True
    
    # Different parameters should return False
    params_new = (np.ones(2), np.ones(2), np.ones(2))
    assert check_convergence(params_new, params, tol=1e-3, criterion="parameters") == False