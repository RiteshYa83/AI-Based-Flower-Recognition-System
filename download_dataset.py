"""Download the public 5-class flower dataset (daisy, dandelion, roses,
sunflowers, tulips; ~3,670 photos, ~220 MB) into data/flower_photos."""
import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATA = Path(__file__).parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    archive = DATA / "flower_photos.tgz"
    if not archive.exists():
        print("Downloading dataset (~220 MB)...")
        urllib.request.urlretrieve(URL, archive)
    print("Extracting...")
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(DATA, filter="data")   # Python 3.12+
        except TypeError:
            tar.extractall(DATA)
    print(f"Done. Train with: python train.py --data-dir {DATA / 'flower_photos'}")


if __name__ == "__main__":
    main()
