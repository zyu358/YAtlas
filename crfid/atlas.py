"""Target images and reference atlases for CRF-ID.

Positions are in micrometres in a common anatomical frame whose axes are
anterior-posterior (AP), left-right (LR) and dorsal-ventral (DV).
"""

import numpy as np

UNLABELED = {"", "?", "na", "n/a", "nan", "none"}


def is_labeled(label):
    """True for a named identity; False for unannotated detections ('?', '', 'NA', ...)."""
    return str(label).strip().lower() not in UNLABELED


def _order(x):
    """o[i, j] = 1 if x[i] > x[j] else 0 for i < j, and o[j, i] = 1 - o[i, j]."""
    n = len(x)
    o = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    up = (x[iu[0]] > x[iu[1]]).astype(float)
    o[iu] = up
    o[iu[1], iu[0]] = 1.0 - up
    return o


class Image:
    """Detected nuclei of one animal and their pairwise features.

    Args:
        id: animal identifier.
        positions: (n, 3) AP, LR, DV coordinates (um).
        labels: n identities ('?' for unannotated nuclei; labels are not used
            for inference, only for scoring and atlas building).

    Attributes:
        ap, lr, dv: (n, n) pairwise order along each axis (see _order).
        vec: (n, n, 3) difference vectors, vec[i, j] = pos[i] - pos[j].
        dist: (n, n) Euclidean distances.
    """

    def __init__(self, id, positions, labels):
        self.id = id
        self.pos = np.array(positions, dtype=float)
        self.labels = [str(l) for l in labels]
        if self.pos.shape != (len(self.labels), 3):
            raise ValueError("positions must be an (n, 3) array with one row per label")
        n = len(self.labels)
        self.ap = _order(self.pos[:, 0])
        self.lr = _order(self.pos[:, 1])
        self.dv = _order(self.pos[:, 2])
        self.vec = self.pos[:, None, :] - self.pos[None, :, :]
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                self.dist[i, j] = self.dist[j, i] = np.linalg.norm(self.vec[i, j])

    def __len__(self):
        return len(self.labels)


class Atlas:
    """Reference statistics of a set of candidate identities over several animals.

    An identity is kept as a candidate (a CRF state) if it is labelled in at
    least `threshold` of the reference images; candidates keep the given order.

    Attributes:
        candidates: (m,) identities.
        mean_pos, pos_sd: (m, 3) mean and SD (ddof 1) of each identity's
            position; pos_sd is NaN for identities seen in one image.
        ap, lr, dv: (m, m) fraction of images in which identity i has the larger
            AP / LR / DV coordinate of the pair (i, j).
        dist, vec: (m, m) mean distance and (m, m, 3) mean difference vector.
        dist_sd: (m, m) SD (ddof 0) of the pairwise distance across images.
        n_obs: (m, m) number of images in which both identities are labelled.
        Pairs never observed together keep ap = lr = dv = dist = 0 and a
        constant direction vector (1, 1, 1) / sqrt(3).
    """

    def __init__(self, images, candidates, threshold=5):
        self.images = list(images)
        index = []
        for im in self.images:
            first = {}
            for k, lab in enumerate(im.labels):
                first.setdefault(lab, k)
            index.append(first)
        kept = [str(c) for c in candidates if sum(str(c) in ix for ix in index) >= threshold]
        self.candidates = np.array(kept, dtype=object)
        m = self.size = len(kept)

        self.mean_pos = np.full((m, 3), np.nan)
        self.pos_sd = np.full((m, 3), np.nan)
        for i, lab in enumerate(kept):
            p = [im.pos[ix[lab]] for im, ix in zip(self.images, index) if lab in ix]
            if p:
                p = np.array(p)
                self.mean_pos[i] = np.mean(p, axis=0)
                if len(p) >= 2:
                    self.pos_sd[i] = np.std(p, axis=0, ddof=1)

        self.ap = np.zeros((m, m))
        self.lr = np.zeros((m, m))
        self.dv = np.zeros((m, m))
        self.dist = np.zeros((m, m))
        self.vec = np.zeros((m, m, 3))
        self.dist_sd = np.zeros((m, m))
        self.n_obs = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                d = []
                for im, ix in zip(self.images, index):
                    r, c = ix.get(kept[i]), ix.get(kept[j])
                    if r is None or c is None:
                        continue
                    self.ap[i, j] += im.ap[r, c]
                    self.lr[i, j] += im.lr[r, c]
                    self.dv[i, j] += im.dv[r, c]
                    self.dist[i, j] += im.dist[r, c]
                    self.vec[i, j] += im.vec[r, c]
                    d.append(im.dist[r, c])
                if d:
                    n = len(d)
                    for a in (self.ap, self.lr, self.dv):
                        a[i, j] /= n
                        a[j, i] = 1.0 - a[i, j]
                    self.dist[i, j] /= n
                    self.dist[j, i] = self.dist[i, j]
                    self.vec[i, j] /= n
                    self.vec[j, i] = -self.vec[i, j]
                    self.n_obs[i, j] = self.n_obs[j, i] = n
                    self.dist_sd[i, j] = self.dist_sd[j, i] = np.std(np.array(d))
                if np.all(self.vec[i, j] == 0.0):
                    self.vec[i, j] = self.vec[j, i] = np.ones(3) / np.sqrt(3.0)
        self.trust = None
        self.trust_scale = None

    def compute_trust(self, scale=None):
        """Edge trust of every identity pair from the variability of its distance.

        trust = exp(-sd^2 / (2 s^2)), with sd the across-animal SD of the pair's
        distance and s = `scale`, by default the median of the positive SDs.
        Pairs observed together in fewer than two images get trust 0.
        """
        observed = self.n_obs >= 2
        if scale is None:
            pos = self.dist_sd[observed & (self.dist_sd > 0)]
            scale = float(np.median(pos)) if pos.size else 1.0
        scale = max(scale, 1e-9)
        trust = np.exp(-(self.dist_sd ** 2) / (2.0 * scale ** 2))
        trust[~observed] = 0.0
        np.fill_diagonal(trust, 0.0)
        self.trust, self.trust_scale = trust, scale
        return trust
