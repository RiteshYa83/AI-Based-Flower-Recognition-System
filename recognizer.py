"""Loads a trained model and predicts flower species for an image."""
import joblib
import numpy as np

from features import extract_features

LOW_CONFIDENCE = 0.50


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


class FlowerRecognizer:
    def __init__(self, model_path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_type = bundle["feature_type"]
        self.classes = list(self.model.classes_)

    def predict(self, image_bgr, top_k: int = 3):
        """Return [(flower_name, probability), ...] sorted best first."""
        feats = extract_features(image_bgr, self.feature_type).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [(pretty(self.classes[i]), float(probs[i])) for i in order]
