import numpy as np
import pytest
from hypervem.utils import match_labels, relabel, find_ambiguous_elements

def test_match_labels_perfect_match():
    """Test that identical labels return an identity permutation."""
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    
    # Expect indices [0, 1, 2]
    perm = match_labels(true, pred)
    np.testing.assert_array_equal(perm, np.array([0, 1, 2]))

def test_match_labels_swapped():
    """Test that swapped labels are correctly identified."""
    true = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])
    
    perm = match_labels(true, pred)
    aligned = relabel(pred, perm)

    np.testing.assert_array_equal(aligned, true)

def test_relabel():
    """Test the application of the permutation."""
    pred = np.array([1, 1, 0, 0, 2])
    # Permutation: map 1->0, 0->1, 2->2 (assuming 3 classes)
    # The perm array indices represent the NEW order.
    # If perm is [1, 0, 2], it means the old index 0 is now at 1, old 1 is at 0.
    perm = np.array([1, 0, 2])
    
    # relabel function: new_labels[pred_labels == new] = old
    # if perm[0] = 1, then pred==1 becomes 0.
    result = relabel(pred, perm)
    expected = np.array([0, 0, 1, 1, 2])
    
    np.testing.assert_array_equal(result, expected)

def test_find_ambiguous_elements():
    """Test detection of low-confidence elements."""
    # 3 items, 2 clusters. 
    # Item 0: 0.95 (Confident)
    # Item 1: 0.55 (Ambiguous)
    # Item 2: 0.80 (Ambiguous)
    resp = np.array([
        [0.95, 0.05],
        [0.55, 0.45],
        [0.20, 0.80]
    ])
    
    df = find_ambiguous_elements(resp, threshold=0.9)
    
    assert len(df) == 2
    assert 1 in df["Index"].values # 0.55 < 0.9
    assert 2 in df["Index"].values # 0.80 < 0.9
    assert 0 not in df["Index"].values