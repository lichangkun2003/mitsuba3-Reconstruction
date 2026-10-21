"""Reproducible RGB8 sRGB PNG export and metrics on the actual saved files.

Both inputs are already composited, linear RGB images. Reference alpha handling
and dataset resampling belong to the dataset reader, not to this module. The
returned PSNR compares decoded PNG bytes in encoded sRGB, without a second
transfer-function decode or foreground mask.
"""

from __future__ import annotations

import math
from pathlib import Path


def _validate_rgb(image, np, label):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or not all(image.shape[:2]):
        raise ValueError(f"{label} must be a nonempty H x W x 3 image; got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{label} contains nonfinite values")
    return image


def _linear_rgb8(image, np):
    # Conversion is explicit rather than delegated to the image writer. The
    # +0.5 then floor rule matches round-to-nearest RGB8 export (ties up), not
    # numpy.rint's ties-to-even or uint8 truncation.
    clipped = np.clip(image, 0, 1)
    srgb = np.where(clipped <= 0.0031308, 12.92 * clipped,
                    1.055 * clipped ** (1 / 2.4) - 0.055)
    return np.floor(np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def read_png_metrics(prediction_path, reference_path, np):
    """Read two saved RGB8 PNGs and return full-image sRGB MSE and PSNR.

    Perfect matches have infinite PSNR, represented by a JSON-safe ``None``
    plus ``perfect_match=True``. Palette/alpha/grayscale inputs are rejected
    to avoid an implicit image conversion changing the evaluation protocol.
    """
    from PIL import Image

    pixels = []
    for label, path in (("Prediction", prediction_path), ("Reference", reference_path)):
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"{label} must be an RGB8 PNG: {path}")
            # Pillow reads raw PNG samples here; it does not apply gamma or
            # color-profile conversion when converting an RGB image to numpy.
            pixels.append(np.asarray(image, dtype=np.float64) / 255.0)
    if pixels[0].shape != pixels[1].shape:
        raise ValueError(f"PNG dimensions differ: {pixels[0].shape} and {pixels[1].shape}")
    mse = float(np.mean((pixels[0] - pixels[1]) ** 2, dtype=np.float64))
    return {"mse": mse, "psnr_db": None if mse == 0 else -10 * math.log10(mse),
            "perfect_match": mse == 0}


def save_srgb_png_pair(prediction_path, reference_path, prediction_linear,
                       reference_linear, np):
    """Save linear prediction/target as sRGB RGB8 PNGs for subsequent scoring.

    Validation of both images occurs before creating directories or writing
    either file. RGB is clipped to [0, 1], sRGB encoded, then quantized using
    ``floor(srgb * 255 + 0.5)``. These are the exact images being scored.
    """
    from PIL import Image

    prediction = _validate_rgb(prediction_linear, np, "Prediction")
    reference = _validate_rgb(reference_linear, np, "Reference")
    if prediction.shape != reference.shape:
        raise ValueError(f"Image dimensions differ: {prediction.shape} and {reference.shape}")
    prediction_path, reference_path = Path(prediction_path), Path(reference_path)
    if prediction_path.resolve() == reference_path.resolve():
        raise ValueError("Prediction and reference PNG paths must differ")
    for path, image in ((prediction_path, prediction), (reference_path, reference)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_linear_rgb8(image, np)).save(path, format="PNG")


def write_srgb_png_pair(prediction_path, reference_path, prediction_linear,
                        reference_linear, np):
    """Save both PNGs, then compute metrics by actually reopening their files."""
    save_srgb_png_pair(prediction_path, reference_path, prediction_linear, reference_linear, np)
    return read_png_metrics(prediction_path, reference_path, np)
