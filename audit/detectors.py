"""Three kinds of 'inspectors' for a (image, mask) pair. All return a suspicion
score: higher = more likely the mask is wrong.

  blind    sees only the table value (mask volume), never the image.
           This is the analogue of the text-only baseline in NeuroQA.
  rules    classical image-aware checks (boundary-edge agreement, connected components).
  vlm      (see vlm.py) a vision-language model looking at an overlay picture.

Detectors are calibrated on a clean reference set only, then applied to a test set.
"""
import numpy as np
from scipy import ndimage as ndi


def volume(mask):
    return float(mask.sum())


def boundary_edge_strength(img, mask):
    """Mean image-gradient magnitude on the mask boundary. A correct mask sits on a
    real intensity edge, so this is high; a shifted/inflated/shrunk mask sits in
    flat regions or inside the structure, so it drops."""
    grad = ndi.gaussian_gradient_magnitude(img, 1.0)
    boundary = mask & ~ndi.binary_erosion(mask)
    return float(grad[boundary].mean()) if boundary.any() else 0.0


def n_components(mask):
    return int(ndi.label(mask)[1])


def _robust_z(x, ref):
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-9
    return (x - med) / mad


class Calibrated:
    """Fit on clean reference cases; score() on any case."""

    def __init__(self, ref_cases):
        self.ref_logvol = np.array([np.log(volume(m)) for _, m in ref_cases])
        self.ref_edge = np.array([boundary_edge_strength(i, m) for i, m in ref_cases])

    def blind(self, img, mask):
        v = volume(mask)
        if v <= 0:
            return 99.0
        return abs(float(_robust_z(np.log(v), self.ref_logvol)))

    def edge_signed(self, img, mask):
        # original assumption: weak boundary edge is suspicious -> negate z
        return -float(_robust_z(boundary_edge_strength(img, mask), self.ref_edge))

    def edge(self, img, mask):
        # two-sided: on real MRI the labelled boundary is not at the steepest gradient,
        # so a shifted boundary can be *stronger* than normal (see README, dilate result)
        return abs(float(_robust_z(boundary_edge_strength(img, mask), self.ref_edge)))

    def components(self, img, mask):
        return float(n_components(mask) - 1)

    def rules(self, img, mask):
        """Image-aware rule score: worst of edge check and component check."""
        return max(self.edge(img, mask), 3.0 * self.components(img, mask))

    def combined(self, img, mask):
        return max(self.blind(img, mask), self.rules(img, mask))
