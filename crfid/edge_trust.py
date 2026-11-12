"""Edge-trust (variability-aware) CRF-ID.

Extends the baseline model with the across-animal variability stored in the atlas:

  position unary   each nucleus scores each candidate identity by a Gaussian of
                   its position around the identity's atlas mean, with the
                   identity's per-axis SD (Config.position_unary, sharpened by
                   the exponent Config.unary_weight).
  edge trust       each identity pair has a trust t = exp(-sd^2 / 2 s^2) from the
                   SD of its distance across atlas animals (Atlas.compute_trust).
                   The binary score of the pair is scaled by f + (1 - f) t, with
                   f = Config.trust_floor (Config.edge_trust).
  message trust    each binary factor's messages are raised to the power
                   sum(P * T) in [0, 1], the trust averaged over its normalized
                   potential P, so factors over unreliable pairs influence their
                   neighbours less (Config.message_trust).
  sparse graph     each nucleus is linked to its Config.knn nearest nuclei
                   (0 links every pair).

The distance term is the squared relative difference 0.5 ((d_t - d_a) / max(d_t, d_a))^2.
With edge_trust and message_trust off, the model keeps the position unary and
the sparse graph; Fig. 3a used this setting for the NeuroPAL and OpenWorm
reference atlases.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .baseline import unit_vectors
from .factorgraph import Graph


@dataclass
class Config:
    """Settings of edge-trust CRF-ID.

    weights          term weights: AP, LR, DV order, distance, angle
    position_unary   use the position unary (otherwise uniform unaries)
    unary_weight     exponent applied to the normalized position unary
    edge_trust       scale each binary score by trust_floor + (1 - trust_floor) * trust
    trust_floor      fraction of the binary score kept for a pair with trust 0
    message_trust    trust-weighted messages
    knn              link each nucleus to its knn nearest nuclei (0: every pair)
    max_iters        maximum number of loopy belief propagation iterations
    """
    weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    position_unary: bool = True
    unary_weight: float = 1.0
    edge_trust: bool = True
    trust_floor: float = 0.5
    message_trust: bool = True
    knn: int = 10
    max_iters: int = 50


def position_unary(atlas, pos, unary_weight=1.0):
    """Normalized Gaussian likelihood of position `pos` under each candidate identity."""
    like = np.zeros(atlas.size)
    for k in range(atlas.size):
        mu, sd = atlas.mean_pos[k], atlas.pos_sd[k]
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))):
            continue
        sd = np.maximum(sd, 1e-6)
        z = (pos - mu) / sd
        like[k] = np.exp(-0.5 * np.sum(z * z)) / np.prod(sd)
    s = like.sum()
    if s <= 0:
        return np.full(atlas.size, 1.0 / atlas.size)
    like = like / s
    if unary_weight != 1.0:
        like = like ** unary_weight
        like = like / like.sum()
    return like


def edge_potential(target, i, j, atlas, weights, trust=None, trust_floor=0.5, atlas_dirs=None):
    """Binary potential (m x m) of target nuclei i, j; trust (m x m) scales each pair's score."""
    w = np.asarray(weights, dtype=float)
    if atlas_dirs is None:
        atlas_dirs = unit_vectors(atlas.vec)
    t_dist = target.dist[i, j]
    t_dir = target.vec[i, j] / np.linalg.norm(target.vec[i, j])
    cos = atlas_dirs @ t_dir
    denom = np.maximum(np.maximum(t_dist, atlas.dist), 1e-12)
    s = w[0] * (1.0 - np.exp((target.ap[i, j] - atlas.ap) ** 2))
    s = s + w[1] * (1.0 - np.exp((target.lr[i, j] - atlas.lr) ** 2))
    s = s + w[2] * (1.0 - np.exp((target.dv[i, j] - atlas.dv) ** 2))
    s = s + w[3] * (1.0 - np.exp(0.5 * ((t_dist - atlas.dist) / denom) ** 2))
    s = s + w[4] * (1.0 - np.exp((cos - 1.0) ** 2))
    if trust is not None:
        s = s * (trust_floor + (1.0 - trust_floor) * trust)
    pot = np.exp(s)
    np.fill_diagonal(pot, 0.0)
    return pot


def knn_pairs(target, k):
    """Pairs (i, j), i < j, such that j is among the k nearest nuclei of i or vice versa."""
    pairs = set()
    for i in range(len(target)):
        cnt = 0
        for j in np.argsort(target.dist[i]):
            j = int(j)
            if j == i:
                continue
            pairs.add((min(i, j), max(i, j)))
            cnt += 1
            if cnt >= k:
                break
    return pairs


def run_crf(atlas, target, config=None):
    """Identify the nuclei of `target` against `atlas`.

    Args:
        atlas: crfid.Atlas. If atlas.trust is None it is computed with
            Atlas.compute_trust(); set it beforehand to use another trust matrix.
        target: crfid.Image; its positions must be in the atlas frame.
        config: Config (defaults: all term weights 1, position unary, edge and
            message trust on, 10 nearest neighbours, 50 iterations).

    Returns:
        (predicted identity per nucleus, marginals array of shape (n, m)).
    """
    cfg = config or Config()
    m, n = atlas.size, len(target)
    trust = None
    if cfg.edge_trust or cfg.message_trust:
        trust = atlas.trust if atlas.trust is not None else atlas.compute_trust()
    pairs = knn_pairs(target, cfg.knn) if cfg.knn > 0 else None
    dirs = unit_vectors(atlas.vec)

    graph = Graph()
    tags = [f"cell{i}" for i in range(n)]
    for tag in tags:
        graph.rv(tag, m)
    for i in range(n):
        unary = position_unary(atlas, target.pos[i], cfg.unary_weight) if cfg.position_unary \
            else np.full(m, 1.0 / m)
        graph.factor([tags[i]], potential=unary)
        for j in range(i + 1, n):
            if pairs is not None and (i, j) not in pairs:
                continue
            pot = edge_potential(target, i, j, atlas, cfg.weights, trust if cfg.edge_trust else None,
                                 cfg.trust_floor, dirs)
            f = graph.factor([tags[i], tags[j]], potential=pot)
            if cfg.message_trust:
                p = pot / pot.sum() if pot.sum() > 0 else pot
                f.weight = float(min(max(float(np.sum(p * trust)), 0.0), 1.0))
    graph.lbp(normalize=True, max_iters=cfg.max_iters)
    marginals = np.array([b for _, b in graph.rv_marginals()])
    return [str(atlas.candidates[k]) for k in marginals.argmax(axis=1)], marginals
