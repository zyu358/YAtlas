"""Baseline CRF-ID.

The CRF-ID model of Chaudhary et al. (eLife 2021) and Lee et al. (eLife 2025):
each detected nucleus of the target image is a random variable whose states are
the atlas candidates. Unary potentials are uniform. Every pair of nuclei is
linked by a binary factor that scores each pair of candidate identities by its
agreement with the atlas in five features: order along AP, LR and DV, distance,
and direction (angle). Loopy belief propagation gives a marginal distribution
per nucleus, and each nucleus takes its most probable identity.
"""

import numpy as np

from .factorgraph import Graph

DEFAULT_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)   # AP, LR, DV order, distance, angle


def unit_vectors(vec):
    """Normalize the last axis of an array of difference vectors."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def edge_potential(target, i, j, atlas, weights, atlas_dirs=None):
    """Binary potential (m x m) of target nuclei i, j over all identity pairs.

    potential = exp(sum_k w_k (1 - exp(c_k))), with c_k the squared mismatch of
    feature k between the target pair and the atlas pair: order differences for
    AP, LR and DV; 0.5 (d_t - d_a) / max(d_t, d_a)^2 for distance d (the form
    of the original implementation); (cos(theta) - 1)^2 for the angle theta
    between the two difference vectors. The diagonal (both nuclei given the
    same identity) is 0.
    """
    w = np.asarray(weights, dtype=float)
    if atlas_dirs is None:
        atlas_dirs = unit_vectors(atlas.vec)
    t_dist = target.dist[i, j]
    t_dir = target.vec[i, j] / np.linalg.norm(target.vec[i, j])
    cos = atlas_dirs @ t_dir
    s = w[0] * (1.0 - np.exp((target.ap[i, j] - atlas.ap) ** 2))
    s = s + w[1] * (1.0 - np.exp((target.lr[i, j] - atlas.lr) ** 2))
    s = s + w[2] * (1.0 - np.exp((target.dv[i, j] - atlas.dv) ** 2))
    s = s + w[3] * (1.0 - np.exp(0.5 * (t_dist - atlas.dist) / np.maximum(t_dist, atlas.dist) ** 2))
    s = s + w[4] * (1.0 - np.exp((cos - 1.0) ** 2))
    pot = np.exp(s)
    np.fill_diagonal(pot, 0.0)
    return pot


def run_crf(atlas, target, weights=DEFAULT_WEIGHTS, max_iters=10):
    """Identify the nuclei of `target` against `atlas`.

    Args:
        atlas: crfid.Atlas.
        target: crfid.Image; its positions must be in the atlas frame.
        weights: (AP, LR, DV order, distance, angle) term weights.
        max_iters: maximum number of loopy belief propagation iterations.

    Returns:
        (predicted identity per nucleus, marginals array of shape (n, m)).
    """
    m, n = atlas.size, len(target)
    dirs = unit_vectors(atlas.vec)
    graph = Graph()
    tags = [f"cell{i}" for i in range(n)]
    for tag in tags:
        graph.rv(tag, m)
    for i in range(n):
        graph.factor([tags[i]], potential=np.full(m, 1.0 / m))
        for j in range(i + 1, n):
            graph.factor([tags[i], tags[j]], potential=edge_potential(target, i, j, atlas, weights, dirs))
    graph.lbp(normalize=True, max_iters=max_iters)
    marginals = np.array([b for _, b in graph.rv_marginals()])
    return [str(atlas.candidates[k]) for k in marginals.argmax(axis=1)], marginals
