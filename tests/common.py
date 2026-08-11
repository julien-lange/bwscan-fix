import numpy as np
import tifffile


def make_tiff(path, seed=0, size=(96, 128), loc=3000, scale=2500, as_rgb=True):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=loc, scale=scale, size=size)
    img = np.clip(base, 0, 65535).astype(np.uint16)
    if as_rgb:
        img = np.repeat(img[:, :, None], 3, axis=2)
    import os

    os.makedirs(str(path.parent), exist_ok=True)
    tifffile.imwrite(str(path), img, photometric="rgb" if as_rgb else None)
    return path