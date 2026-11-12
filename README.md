# *C. elegans* head-neuron atlas: atlas explorer and variability-aware CRF-ID

Code and data accompanying Yu et al. (manuscript in preparation), which builds a
brain-wide atlas of *C. elegans* head neurons from multi-cell reporter strains
and uses its across-animal variability for automated neuron identification.

| Folder | Contents |
| --- | --- |
| `app/` | `YAtlas-explorer.html`, the interactive atlas explorer (one self-contained file) |
| `crfid/` | Python CRF-ID: baseline and edge-trust (variability-aware) inference |
| `crfid/factorgraph/` | factor graphs and loopy belief propagation, a modified copy of [py-factorgraph](https://github.com/mbforbes/py-factorgraph) (MIT License) |
| `demo/` | leave-one-out demo on 23 *inx-4p & mbr-1p* animals that reproduces their Fig. 3a values |

## Installation

Python 3.9 or later with numpy, pandas, matplotlib and scipy:

```
pip install -r requirements.txt
```

Tested with Python 3.12, numpy 1.26, pandas 2.2, matplotlib 3.9 and scipy 1.13.
`crfid` itself needs only numpy; the demo also uses pandas, matplotlib and scipy.

## Atlas explorer

Open `app/YAtlas-explorer.html` in Chrome or Edge. All atlas data are embedded
in the file, nothing is uploaded, and it also works offline. It shows the
five-strain atlas (version `glr1_acr5_inx4_ncs1_odr2_robust_20260624`: 125 cells
from 96 animals) as a graph: cells are nodes with their positional variability,
and pairwise relations are edges shaded by their stability across animals
(darker = more stable). Click a cell or an edge for its statistics; tables,
statistics and images can be downloaded as CSV, JSON and PNG. The "Identify your
neurons" buttons describe the full identification workflow, which needs the
server version of the app and is not part of this file.

## CRF-ID

Each detected nucleus of a target animal is a random variable whose states are
the candidate identities of an atlas. Loopy belief propagation gives each nucleus
a marginal distribution over identities, and the nucleus takes the most probable
one. All positions are in micrometres in the atlas frame (anterior-posterior,
left-right and dorsal-ventral axes), so the target must first be aligned to the
atlas.

**Baseline** (`crfid/baseline.py`). A Python implementation of CRF-ID
(Chaudhary et al., 2021; Lee et al., 2025). Unary potentials are uniform and
every pair of nuclei (a, b) has a binary factor. For a candidate identity pair
(i, j) the factor is

    psi = exp( sum_k w_k (1 - exp(c_k)) )

with five mismatch terms c_k between the target pair and the atlas pair: the
order along AP, LR and DV, `(o_ab - p_ij)^2`, where `o_ab` is 1 if a has the
larger coordinate and `p_ij` is the fraction of atlas animals in which i has the
larger coordinate; the distance, `0.5 (d_ab - d_ij) / max(d_ab, d_ij)^2` (the
form of the original implementation); and the angle between the two difference
vectors, `(cos theta - 1)^2`. A pair of nuclei cannot take the same identity
(psi = 0). Loopy belief propagation runs for up to 10 iterations.

**Edge trust** (`crfid/edge_trust.py`), the variability-aware CRF-ID of the
paper. It adds the atlas variability to the baseline:

- *Position unary*: a nucleus at position x scores identity i by
  `prod_axes N(x; mu_i, sd_i)`, normalized over identities and raised to the
  power `unary_weight`.
- *Edge trust*: each identity pair gets `T_ij = exp(-sd_ij^2 / (2 s^2))`, where
  `sd_ij` is the across-animal SD of its distance and s is the median of the
  positive SDs; pairs seen together in fewer than two atlas animals get T = 0.
  The binary score of the pair is multiplied by `f + (1 - f) T_ij`
  (`trust_floor` f = 0.5).
- *Trust-weighted messages*: each binary factor raises its outgoing messages to
  the power `sum_ij P_ij T_ij` (its normalized potential P averaged against T),
  so factors over unreliable relations influence their neighbours less.
- *Sparse graph*: each nucleus is linked to its 10 nearest nuclei (`knn`).
- The distance term is the squared relative difference
  `0.5 ((d_ab - d_ij) / max(d_ab, d_ij))^2`, and loopy belief propagation runs
  for up to 50 iterations.

Every component can be switched off in `edge_trust.Config`. The defaults of both
programs give every term weight 1.

```python
from crfid import Atlas, Image, baseline, edge_trust

# reference animals in the atlas frame: positions (n x 3, um) and identities
refs = [Image(name, pos, labels) for name, pos, labels in reference_animals]
atlas = Atlas(refs, candidates=candidate_identities, threshold=5)

target = Image("worm", aligned_positions, ["?"] * len(aligned_positions))
pred, marginals = edge_trust.run_crf(atlas, target)          # all weights 1
cfg = edge_trust.Config(weights=(1, 1, 1, 0.2, 5), knn=10)
pred, marginals = edge_trust.run_crf(atlas, target, cfg)
pred, marginals = baseline.run_crf(atlas, target, weights=(1, 1, 1, 1, 1))
```

`pred` lists the predicted identity of each nucleus and `marginals` (nuclei x
candidates, columns ordered as `atlas.candidates`) gives the full distributions,
from which ranked candidate lists can be read.


## Third-party code and data

- `crfid/factorgraph/` is a modified copy of py-factorgraph 0.0.3 by Maxwell
  Forbes (https://github.com/mbforbes/py-factorgraph), used under the MIT License;
  the license text is in `crfid/factorgraph/LICENSE`. Changes: (1) messages from
  a variable or a factor are formed as the product over the other incoming
  messages instead of dividing the full product by one message; (2) messages
  and beliefs are renormalized after every multiplication, and a product that
  underflows to zero is replaced by a uniform vector; (3) a per-factor weight in
  [0, 1] to which outgoing messages are raised (trust-weighted messages);
  (4) the Ctrl-C handler installed at import time was removed. The module
  docstring lists the same changes.
- The NeuroPAL atlas is derived from DANDI:000715 (Yemini et al., 2024), released
  under CC-BY-4.0.
- The OpenWorm coordinates come from the OpenWorm project (Szigeti et al., 2014).

## References

- Chaudhary S, Lee SA, Li Y, Patel DS, Lu H. Graphical-model framework for
  automated annotation of cell identities in dense cellular images. *eLife* 10,
  e60321 (2021). https://doi.org/10.7554/eLife.60321
- Lee HJ, Liang J, Chaudhary S, Moon S, Yu Z, Wu T, Liu H, Choi M-K, Zhang Y,
  Lu H. Automated cell annotation in multi-cell images using an improved CRF_ID
  algorithm. *eLife* 12, RP89050 (2025). https://doi.org/10.7554/eLife.89050
- Forbes M. py-factorgraph: factor graphs and loopy belief propagation.
  https://github.com/mbforbes/py-factorgraph
- Yemini E, Lin A, Nejatbakhsh A, Varol E, Sun R, Mena GE, Samuel ADT,
  Paninski L, Venkatachalam V, Hobert O. NeuroPAL: a multicolor atlas for
  whole-brain neuronal identification in *C. elegans*. *Cell* 184, 272-288
  (2021). https://doi.org/10.1016/j.cell.2020.12.012
- Yemini E, Venkatachalam V, Lin A, Varol E, Sprague D, Paninski L, Samuel A,
  Hobert O. NeuroPAL: Atlas of *C. elegans* neuron locations and colors in
  NeuroPAL worm (version 0.241009.1514). DANDI archive (2024).
  https://doi.org/10.48324/dandi.000715/0.241009.1514
- Szigeti B, Gleeson P, Vella M, Khayrulin S, Palyanov A, Hokanson J, Currie M,
  Cantarelli M, Idili G, Larson S. OpenWorm: an open-science approach to
  modeling *Caenorhabditis elegans*. *Front Comput Neurosci* 8, 137 (2014).
  https://doi.org/10.3389/fncom.2014.00137

## License

`crfid/factorgraph/` is distributed under its own MIT License
(`crfid/factorgraph/LICENSE`). See `LICENSE` for the rest of this repository.
