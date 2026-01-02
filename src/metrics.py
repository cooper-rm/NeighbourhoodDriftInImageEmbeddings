# src/metrics.py
import numpy as np



def compute_overlap_stats(
    exact_neighbors,
    ann_neighbors, 
    k
):
    
    """
    Compute mean and std of neighborhood overlap.

    Parameters
    ----------
    exact_neighbors : list[list[int]]
        Exact k-NN indices for each query.
    ann_neighbors : list[list[int]]
        Approximate k-NN indices for each query.
    k : int
        Number of neighbors.

    Returns
    -------
    mean_overlap : float
    std_overlap : float
    """
    overlaps = []

    for exact, ann in zip(exact_neighbors, ann_neighbors):
        exact_set = set(exact)
        ann_set = set(ann)
        overlap = len(exact_set & ann_set) / k
        overlaps.append(overlap)

    overlaps = np.asarray(overlaps)
    return overlaps.mean(), overlaps.std()




def compute_distance_divergence_stats(
    D_exact, 
    D_ann
):
    
    """
    Compute normalized divergence between ANN and exact k-NN distances.

    Parameters
    ----------
    D_exact : np.ndarray, shape (n_queries, k)
        Exact k-NN distances
    D_ann : np.ndarray, shape (n_queries, k)
        ANN k-NN distances

    Returns
    -------
    mean_divergence : float
    std_divergence : float
    """
    
    # Mean distance per query
    mean_exact = np.mean(D_exact, axis=1)
    mean_ann = np.mean(D_ann, axis=1)

    # Normalized divergence per query
    eps = 1e-12
    divergence = (mean_ann - mean_exact) / (mean_exact + eps)


    return float(np.mean(divergence)), float(np.std(divergence))




def compute_barycenter_stats(
    embeddings, 
    I_exact, 
    I_ann, 
    D_exact
):
    
    """
    Normalized barycenter shift between exact and ANN neighborhoods.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_points, d)
        Embedding vectors
    I_exact : np.ndarray, shape (n_queries, k)
        Exact neighbor indices
    I_ann : np.ndarray, shape (n_queries, k)
        ANN neighbor indices
    D_exact : np.ndarray, shape (n_queries, k)
        Exact neighbor distances (used to compute radius)

    Returns
    -------
    mean_shift : float
    std_shift : float
    """
    
    shifts = []

    for i in range(len(I_exact)):
        # Neighbor embeddings
        exact_neighbors = embeddings[I_exact[i]]
        ann_neighbors = embeddings[I_ann[i]]

        # Barycenters
        c_exact = exact_neighbors.mean(axis=0)
        c_ann = ann_neighbors.mean(axis=0)

        # Absolute shift
        shift = np.linalg.norm(c_exact - c_ann)

        # Neighborhood radius (mean exact distance)
        radius = np.mean(D_exact[i])

        # Normalize (guard against zero radius)
        if radius > 0:
            shifts.append(shift / radius)
        else:
            shifts.append(0.0)

    shifts = np.asarray(shifts)
    return shifts.mean(), shifts.std()



def compute_lid(distances, eps=1e-12):
    """
    Compute LID for a single query given its k-NN distances.

    Parameters
    ----------
    distances : np.ndarray, shape (k,)
        Sorted distances to k nearest neighbors
    eps : float
        Numerical stability constant

    Returns
    -------
    lid : float
    """
    d_k = distances[-1] + eps
    ratios = distances[:-1] / d_k
    return -1.0 / np.mean(np.log(ratios + eps))



def compute_lid_stats(D_exact, D_ann, eps=1e-12):
    """
    Compute LID statistics for exact vs ANN neighborhoods.

    Returns
    -------
    stats : dict
        {
            "mean_lid_diff": float,
            "std_lid_diff": float,
            "mean_lid_exact": float,
            "mean_lid_ann": float,
        }
    """

    # Per-query LID
    lids_exact = np.array([compute_lid(D_exact[i], eps) for i in range(len(D_exact))])
    lids_ann   = np.array([compute_lid(D_ann[i],   eps) for i in range(len(D_ann))])

    # LID difference (ANN − exact)
    lid_diff = lids_ann - lids_exact

    return {
        "mean_lid_diff": float(lid_diff.mean()),
        "std_lid_diff":  float(lid_diff.std()),
        "mean_lid_exact":      float(lids_exact.mean()),
        "mean_lid_ann":        float(lids_ann.mean()),
    }
