# MNIST Digit Recognizer App

A Streamlit web application for interactive handwritten digit recognition using a pre-trained CNN model.

## What it does

- Provides a drawing canvas for users to draw digits (0-9)
- Preprocesses the drawn image (grayscale conversion, thresholding, contour detection, cropping, resizing)
- Uses a TensorFlow/Keras CNN model to predict the digit
- Displays the processed image and prediction result

## Requirements

- Python 3.x
- `streamlit`
- `numpy`
- `Pillow`
- `tensorflow`
- `opencv-python`
- `streamlit-drawable-canvas`

## Model File

- `cnn_mnist.keras` — Pre-trained CNN model for digit recognition

## Installation

Install dependencies with:

```bash
pip install streamlit numpy Pillow tensorflow opencv-python streamlit-drawable-canvas
```

## Usage

Run the app with:

```bash
streamlit run app.py
```

Then open the provided URL in your browser to use the app.

## Notes

- The CNN model should be trained separately (see `cnn.py` for training code if available).
- The app uses `streamlit-drawable-canvas` for the drawing interface.
- Images are processed to 28x28 pixels before prediction.

## Files

- `app.py` — Main Streamlit application
- `cnn_mnist.keras` — Pre-trained model file

## License

This repository does not include a license file. Add one if you plan to share or publish the code.
