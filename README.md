# MNIST Training Script

This repository contains a simple Python script, `mnist_train.py`, that trains and evaluates several machine learning models on the MNIST handwritten digit dataset.

## What it does

`mnist_train.py` performs the following steps:

- Downloads the `mnist_784` dataset from OpenML (if not already cached)
- Normalizes pixel values to the range `[0, 1]`
- Splits the data into training and test sets
- Trains and evaluates:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Applies PCA to reduce dimensionality and trains Logistic Regression again
- Prints the accuracy of each model and ranks the results

## Requirements

- Python 3.x
- `scikit-learn`

Install dependencies with:

```bash
pip install scikit-learn
```

## Usage

Run the script from the repository folder:

```bash
python mnist_train.py
```

The script may take several minutes the first time it runs because it needs to download the MNIST dataset.

## Notes

- If the dataset download fails, the script prints an error message with details.
- The script uses `train_test_split` with `random_state=42` for reproducible results.
- PCA is applied with `n_components=50` before retraining Logistic Regression.

## Files

- `mnist_train.py` — main training and evaluation script

## License

This repository does not include a license file. Add one if you plan to share or publish the code.
