"""CRF-ID: identification of cells in a target image against a positional atlas.

    baseline     CRF-ID with uniform unaries and five weighted binary terms
    edge_trust   variability-aware CRF-ID: position unary, edge trust and
                 trust-weighted messages
    factorgraph  factor graphs and loopy belief propagation (modified
                 py-factorgraph; see factorgraph/LICENSE)
"""

from . import baseline, edge_trust
from .atlas import Atlas, Image, is_labeled

__all__ = ["Atlas", "Image", "is_labeled", "baseline", "edge_trust"]
