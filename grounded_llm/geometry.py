"""Representation diagnostics use the existing map geometry authority."""
import numpy as np
from experiments.cml_map_scaling.src.core import latent_distances, spearman

def geometry_stat(vectors, distances):
    i, j = np.triu_indices(len(vectors), 1)
    return {'pairs': len(i), 'spearman': spearman(distances[i, j], latent_distances(vectors)[i, j])}
