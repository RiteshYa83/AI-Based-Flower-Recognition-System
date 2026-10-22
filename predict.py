"""Command-line prediction.  Usage: python predict.py photo1.jpg photo2.jpg"""
import argparse
from pathlib import Path

from features import read_image_path
from recognizer import LOW_CONFIDENCE, FlowerRecognizer

BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(BASE / "models" / "flower_model.joblib"))
    args = ap.parse_args()

    rec = FlowerRecognizer(args.model)
    for path in args.images:
        try:
            top = rec.predict(read_image_path(path))
        except Exception as exc:
            print(f"{path}: error - {exc}")
            continue
        name, prob = top[0]
        note = "  (low confidence)" if prob < LOW_CONFIDENCE else ""
        print(f"{path}: {name} ({prob:.1%}){note}")
        for n, p in top[1:]:
            print(f"    also possible: {n} ({p:.1%})")


if __name__ == "__main__":
    main()
