import numpy as np
from sklearn.cluster import AgglomerativeClustering

def create_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray:
    """
    Creates linkage matrix representing the agglomerative clustering on the fit model

    Args:
        model - Agglomerative clustering model fit on the data

    Returns:
        np.ndarray - linkage matrix representing the data in the agglomerative clustering
    """

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix