import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import pandas as pd


def match_labels(true_labels, pred_labels):
    """
    Finds the optimal permutation of predicted labels to match true labels
    using the Hungarian algorithm (Trace Maximization of Confusion Matrix).

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth label assignments.

    pred_labels : np.ndarray
        Inferred label assignments from the model.

    Returns
    -------
    np.ndarray
        An index array used to reorder the model parameters so they match the ground truth order.
    """
    valid_mask = ~np.isnan(true_labels) & ~np.isnan(pred_labels)

    if np.sum(valid_mask) == 0:
        # Fallback if no valid data
        return np.arange(len(np.unique(pred_labels)))
    
    # Compute Confusion Matrix
    cm = confusion_matrix(true_labels[valid_mask], pred_labels[valid_mask])

    # Find optimal assignment (maximum trace)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Sort indices to align with ground truth
    reordered_indices = col_ind[np.argsort(row_ind)]

    return reordered_indices


def relabel(pred_labels, perm):
    """
    Permutes the predicted labels to align with the ground truth indices.

    Parameters
    ----------
    pred_labels : np.ndarray
        The original predicted label assignments.

    perm : np.ndarray
        The optimal permutation index array (output of match_labels).

    Returns
    -------
    np.ndarray
        The permuted predicted labels, re-indexed to match the true labels.
    """
    new_labels = np.zeros_like(pred_labels)
    for old, new in enumerate(perm):
        new_labels[pred_labels == new] = old
    return new_labels

def find_ambiguous_elements(responsibilities, threshold):
    """
    Identifies elements where the model's highest cluster probability
    is below a specified confidence threshold.

    Parameters
    ----------
    responsibilities : np.ndarray
        The responsibilities vector, which contains the probability
        of each element belonging to each cluster.

    threshold : float, optional (default=0.9)
        The confidence cutoff. Elements with a max cluster probability below this
        value are considered ambiguous.

    Returns
    -------
    tuple of pd.DataFrame
        - ambiguous_elements: DataFrame containing indices and probabilities of ambiguous elements.
    """
    # Find the highest probability for each row
    max_probs = np.max(responsibilities, axis = 1)
    # Flag low confidence 
    ambiguous_indices = np.where(max_probs < threshold)[0]
    ambiguous_probs = max_probs[ambiguous_indices]
    # Create a sorted DataFrame (lowest confidence first)
    ambiguous_elements = pd.DataFrame(
            {"Index": ambiguous_indices, "Confidence": ambiguous_probs}
        ).sort_values("Confidence", ascending=True)

    return ambiguous_elements