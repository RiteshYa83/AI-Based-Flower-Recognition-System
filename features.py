"""Image loading and feature extraction.

Two feature extractors are available:

* "classic"   - HSV colour histogram + HOG (shape) + LBP (texture).
                Runs anywhere, no deep-learning library needed.
* "mobilenet" - MobileNetV2 (ImageNet weights) as a frozen feature extractor.
                Needs TensorFlow; usually much more accurate on real photos.
"""
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = 128
FEATURE_TYPES = ("classic", "mobilenet")
_mobilenet = None


def read_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file could not be read as an image.")
    return img


def read_image_path(path) -> np.ndarray:
    return read_image_bytes(Path(path).read_bytes())


def _classic_features(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Colour: HSV histogram (hue gets the most bins - colour is key for flowers)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4],
                        [0, 180, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-9

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shape: histogram of oriented gradients
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm="L2-Hys")

    # Texture: uniform local binary patterns
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-9)

    return np.concatenate([hist, shape, lbp_hist]).astype(np.float32)


def _mobilenet_features(img: np.ndarray) -> np.ndarray:
    global _mobilenet
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    if _mobilenet is None:
        _mobilenet = MobileNetV2(weights="imagenet", include_top=False,
                                 pooling="avg", input_shape=(224, 224, 3))
    rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    x = preprocess_input(rgb.astype("float32")[None])
    return _mobilenet.predict(x, verbose=0)[0].astype(np.float32)


def extract_features(img: np.ndarray, feature_type: str = "classic") -> np.ndarray:
    if feature_type == "classic":
        return _classic_features(img)
    if feature_type == "mobilenet":
        return _mobilenet_features(img)
    raise ValueError(f"Unknown feature type: {feature_type!r}")
