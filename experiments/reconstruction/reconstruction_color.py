"""Reference-image compositing conventions; returned targets are always linear.

The ``srgb`` convention matches the encoded PNG/alpha compositing used by
NeRF-Synthetic benchmarks, then decodes the composited color for this renderer's
linear-space optimization. Already-linear sources (for example EXR) retain
linear compositing under either convention.
"""

from __future__ import annotations


def validate_reference_compositing(value):
    if not isinstance(value, str) or value not in ("linear", "srgb"):
        raise ValueError("data.reference_compositing must be 'linear' or 'srgb'")


def srgb_to_linear(values, np):
    """Decode sRGB RGB values without clipping HDR values or applying gamma to alpha."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.maximum((values + 0.055) / 1.055, 0) ** 2.4,
    ).astype(np.float32, copy=False)


def linear_to_srgb(values, np):
    """Encode linear RGB with the sRGB transfer function, without clamping."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def reference_from_bitmap(bitmap, mi, np, *, rgb_only=False, background=(1, 1, 1),
                          reference_compositing="linear", width=None, height=None,
                          path=None):
    """Return owned float32 linear RGBA, or composited linear RGB.

    Alpha remains linear coverage throughout. Bitmap conversion supplies alpha
    one for ordinary RGB files. Only semitransparent pixels from sRGB-tagged
    inputs need the encoded-domain blend; opaque pixels retain Bitmap's original
    linear decoding exactly, and transparent pixels equal the linear background.
    """
    validate_reference_compositing(reference_compositing)
    linear = bitmap.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, False)
    rgba = np.array(linear, dtype=np.float32, order="C", copy=True)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"Expected an RGBA reference, got dimensions {rgba.shape}: {path}")
    if width is not None or height is not None:
        if rgba.shape != (height, width, 4):
            raise ValueError(
                f"Reference dimensions {rgba.shape} differ from {(height, width, 4)}: "
                f"{path}; no implicit resampling")
    if not np.isfinite(rgba).all() or np.any(rgba[..., 3] < 0) or np.any(rgba[..., 3] > 1):
        raise ValueError(f"Reference contains nonfinite values or invalid alpha: {path}")
    if not rgb_only:
        # Legacy alpha-supervised callers consume the original linear RGBA.
        return rgba

    background = np.asarray(background, dtype=np.float32)
    if background.shape != (3,) or not np.isfinite(background).all():
        raise ValueError("Reference background must contain three finite values")
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + background * (1 - alpha)
    partial = (alpha[..., 0] > 0) & (alpha[..., 0] < 1)
    if reference_compositing == "srgb" and bitmap.srgb_gamma() and np.any(partial):
        encoded = np.array(
            bitmap.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, True),
            dtype=np.float32, copy=True)
        composite = (encoded[..., :3][partial] * alpha[partial]
                     + linear_to_srgb(background, np) * (1 - alpha[partial]))
        rgb[partial] = srgb_to_linear(composite, np)
    if not np.isfinite(rgb).all():
        raise ValueError(f"Reference compositing produced nonfinite values: {path}")
    # Do not let the RGB-only cache retain a hidden RGBA backing allocation.
    return np.array(rgb, dtype=np.float32, order="C", copy=True)
