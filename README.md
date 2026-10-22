# Flower Recognition System

Identify flower species from an uploaded photo using Python and machine learning.
Upload an image in the web app and it shows the predicted flower name, the confidence, and the top 3 candidates.

## Project structure

    flower_recognition/
    ├── app.py                # Flask web app (upload page + JSON API)
    ├── train.py              # trains and evaluates the classifier
    ├── predict.py            # command-line prediction
    ├── features.py           # image loading + feature extraction
    ├── recognizer.py         # loads the saved model and predicts
    ├── download_dataset.py   # downloads a public 5-species flower dataset
    ├── make_demo_data.py     # tiny synthetic dataset, for a quick pipeline check only
    ├── templates/index.html  # web page
    └── requirements.txt

## Quick start

    pip install -r requirements.txt
    python download_dataset.py            # daisy, dandelion, roses, sunflowers, tulips
    python train.py --data-dir data/flower_photos
    python app.py                         # open http://127.0.0.1:5000

Training writes `models/flower_model.joblib`, `models/confusion_matrix.png` and `models/metrics.json`.

## Use your own flowers

Create one folder per species, with the photos inside, then point `--data-dir` at it:

    my_flowers/
        lotus/   *.jpg
        jasmine/ *.jpg
        marigold/ *.jpg

    python train.py --data-dir my_flowers

The folder names become the predicted labels. Aim for 100+ photos per species.

## How it works

1. Each image is converted to a feature vector.
2. Three classifiers (SVM, Random Forest, Logistic Regression) are compared with cross-validation on the training split.
3. The best one is evaluated on a held-out 20% test set and saved.
4. The web app and `predict.py` load the saved model and return class probabilities.

Two feature extractors are available (`--features`):

| Option | What it uses | Notes |
|---|---|---|
| `classic` (default) | HSV colour histogram + HOG shape + LBP texture | Works with the installed requirements, no deep learning |
| `mobilenet` | MobileNetV2 (ImageNet weights) as a frozen feature extractor | Needs `pip install tensorflow` and internet on first run (downloads weights); usually much more accurate on real photos |

To get the best accuracy, use:

    pip install tensorflow
    python train.py --data-dir data/flower_photos --features mobilenet

## Command-line prediction

    python predict.py rose.jpg tulip.png

## JSON API

    curl -F image=@rose.jpg http://127.0.0.1:5000/api/predict

returns

    {"prediction": "Rose", "confidence": 0.91, "low_confidence": false,
     "top3": [{"flower": "Rose", "confidence": 0.91}, ...]}

## Notes and limitations

- The model only knows the species it was trained on. A photo of anything else is still assigned to the closest known species, so the app shows a warning when confidence is below 50%.
- The `classic` extractor is a lightweight baseline. Expect noticeably lower accuracy on real photos than with `mobilenet`.
- `make_demo_data.py` draws synthetic flowers purely to check the code runs. Do not use a model trained on it for real photos.
- Uploads are limited to 8 MB (JPG, PNG, BMP, WEBP). They are not saved to disk.
- The web server is for local use. For deployment, run it behind a production server such as gunicorn.
