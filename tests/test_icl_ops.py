import numpy as np
import pytest
from hypervem.icl_ops import compute_penalty, compute_entropy

def test_penalty_computation():
    """Test BIC penalty formula."""
    # K=2, G=2. Free params = KG + (K-1) + (G-1) = 4 + 1 + 1 = 6
    # Penalty = 0.5 * 6 * log(MN)
    MN = 100
    expected = 0.5 * 6 * np.log(MN)
    assert compute_penalty(MN, 2, 2) == expected

def test_entropy_computation():
    """Test entropy is near zero for almost-deterministic distribution."""
  
    resp_safe = np.array([[0.9999, 0.0001]])
    ent = compute_entropy(resp_safe)
    assert ent < 0.01 # Should be very small