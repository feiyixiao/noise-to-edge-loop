"""Render (image, mask) as a picture a vision-language model can look at:
three orthogonal slices through the mask centre, grayscale MRI, mask outline in red.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PANEL = 224  # pixels per panel


def _panel(slice2d, mask2d):
    g = (np.clip(slice2d, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    edge = mask2d & ~ndi.binary_erosion(mask2d)
    rgb[edge] = (255, 0, 0)
    im = Image.fromarray(rgb)
    # nearest-neighbour keeps the one-voxel outline sharp
    return im.resize((PANEL, PANEL), Image.NEAREST)


def render_v2(img, mask, outline=True):
    """Three orthogonal slices through the centroid of the largest connected piece of the mask
    (what a reviewer would pick; no knowledge of where any damage is) + one panel showing the
    mask alone, projected along axis 0, so a distant extra piece is visible."""
    lab, n = ndi.label(mask)
    if n > 1:
        keep = 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
        big = lab == keep
    else:
        big = mask
    c = [int(round(x)) for x in ndi.center_of_mass(big)] if big.any() else [s // 2 for s in img.shape]
    shown = mask if outline else np.zeros_like(mask)   # outline=False: same slices, no red line, blank projection
    panels = [
        _panel(img[c[0], :, :], shown[c[0], :, :]),
        _panel(img[:, c[1], :], shown[:, c[1], :]),
        _panel(img[:, :, c[2]], shown[:, :, c[2]]),
    ]
    proj = shown.any(axis=0)
    g = (proj * 255).astype(np.uint8)
    panels.append(Image.fromarray(np.stack([g, g, g], -1)).resize((PANEL, PANEL), Image.NEAREST))
    out = Image.new("RGB", (PANEL * 4 + 12, PANEL), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL + 4), 0))
    return out


def render(img, mask):
    if mask.any():
        c = [int(round(x)) for x in ndi.center_of_mass(mask)]
    else:
        c = [s // 2 for s in img.shape]
    panels = [
        _panel(img[c[0], :, :], mask[c[0], :, :]),
        _panel(img[:, c[1], :], mask[:, c[1], :]),
        _panel(img[:, :, c[2]], mask[:, :, c[2]]),
    ]
    out = Image.new("RGB", (PANEL * 3 + 8, PANEL), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL + 4), 0))
    return out
