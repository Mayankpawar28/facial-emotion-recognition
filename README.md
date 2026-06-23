# Facial Emotion Recognition

A Convolutional Neural Network (CNN) that detects human emotions from grayscale face images, trained on the FER2013 dataset. Classifies faces into **7 emotion categories** using TensorFlow and Keras.

---

## Emotions Detected

| Label | Emotion |
|---|---|
| 0 | Angry 😠 |
| 1 | Disgust 🤢 |
| 2 | Fear 😨 |
| 3 | Happy 😄 |
| 4 | Neutral 😐 |
| 5 | Sad 😢 |
| 6 | Surprise 😲 |

---

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Kaggle API (`kaggle`)
- Google Colab

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib kaggle
```

---

## Setup

### 1. Kaggle API Key
Upload your `kaggle.json` API key when prompted in Colab, then run:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset

```bash
kaggle datasets download -d msambare/fer2013
unzip -q fer2013.zip
```

The dataset extracts into `train/` and `test/` folders, each containing one subfolder per emotion class.

---

## Pipeline

### Step 1 — Preprocess Images
- Rescale pixel values to `[0, 1]`
- Apply augmentation on training data: rotation, zoom, horizontal flip
- Resize all images to **48×48 pixels** (grayscale)
- Batch size: **64**

### Step 2 — Build CNN Model

| Layer | Details |
|---|---|
| Conv2D + MaxPooling | 32 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 64 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 128 filters, 3×3, ReLU |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | 7 units, Softmax |

### Step 3 — Compile & Train
- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam
- **Epochs:** 10

### Step 4 — Evaluate
Prints test loss and accuracy after training.

### Step 5 — Predict on a New Image
Loads a face image, preprocesses it, and outputs the predicted emotion with confidence score.

### Step 6 — Visualize Training
Plots training vs. validation accuracy over epochs using Matplotlib.

---

## Example Output

```
Predicted Emotion: Happy
Confidence: 94.32 %
```

---

## Dataset

**Source:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) by msambare

**Size:** ~35,000 grayscale images (48×48 px) across 7 emotion classes
