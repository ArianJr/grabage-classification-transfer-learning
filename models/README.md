# Saved Models

This directory contains references to the trained model checkpoints used in this project.  
Due to file size limitations, the actual weights are hosted externally.

## 📂 Download All Models
All trained models are stored in a single Google Drive folder:  
[Access Models Here](https://drive.google.com/drive/folders/1vJXIkJczFYWPh-UUW8k6wE00_9qX9FH-?usp=drive_link)

## 🧠 Included Architectures

This project benchmarks four convolutional neural network architectures for garbage classification:

- **CNN from Scratch** – trained end-to-end without pretrained weights.
- **ResNet50** – pretrained on ImageNet, fine-tuned on garbage dataset.
- **MobileNetV2** – pretrained on ImageNet, fine-tuned on garbage dataset.
- **EfficientNetB0** – pretrained on ImageNet, fine-tuned on garbage dataset.

## ⚙️ Usage
After downloading, place the `.h5` files in the `models/` directory before running evaluation scripts.

```python
from tensorflow.keras.models import load_model

model = load_model("models/resnet50.h5")
```

## 📌 Notes
- All checkpoints are provided in `.h5` format for compatibility with TensorFlow/Keras.
- Models were trained and evaluated on the same dataset for fair comparison.
- Due to file size limitations, weights are hosted externally (Google Drive).
- Place downloaded files in the `models/` directory before running evaluation scripts.
- Filenames follow a consistent convention (e.g., `cnn_scratch_best.h5`, `resnet50_best.h5`, `mobilenetv2_best.h5`, `efficientnetb0_best.h5`).
