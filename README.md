# Garbage Classification with Transfer Learning

---

## Overview

This project delivers a **comprehensive and visually grounded evaluation** of deep learning approaches for **image-based garbage classification**. The study compares a baseline convolutional neural network trained from scratch against multiple **state-of-the-art transfer learning architectures**, all evaluated under identical experimental conditions.

All experiments, visualizations, and analyses are contained **directly within the notebook**, ensuring transparency, reproducibility, and clarity.

---

## Model Architectures

### CNN from Scratch

Baseline convolutional neural network trained without pretrained weights. This model establishes a lower-bound reference for performance and learning behavior.

### ResNet50 (Transfer Learning)

Deep residual network pretrained on ImageNet, used as a feature extractor to leverage rich hierarchical representations.

### MobileNetV2 (Transfer Learning)

Lightweight architecture optimized for efficiency while maintaining strong predictive performance. Suitable for deployment-oriented scenarios.

### EfficientNetB0 (Transfer Learning)

Modern architecture using compound scaling to balance depth, width, and resolution, achieving high accuracy with fewer parameters.

### Comparative Analysis

All models are evaluated side by side to highlight architectural trade-offs and performance trends.

---

## Experimental Configuration

| Component           | Description                                    |
| ------------------- | ---------------------------------------------- |
| Input Resolution    | Uniform resizing across all models             |
| Loss Function       | Categorical Crossentropy                |
| Optimizer           | Adam                                           |
| Evaluation Metric   | Accuracy                                       |
| Data Augmentation   | Random flips, rotations, zooms (training only) |
| Validation Strategy | Held-out validation dataset                    |

This controlled setup ensures a fair and meaningful comparison across architectures.

---

## Visual Evaluation

### Training and Validation Curves

The notebook includes **five complete sets of loss and accuracy curves**:

* CNN from Scratch
* ResNet50
* MobileNetV2
* EfficientNetB0
* Combined cross-model comparison

These visualizations provide insight into:

* Convergence speed
* Generalization behavior
* Overfitting tendencies

Maintaining all curves supports transparent and evidence-based model selection.


### Confusion Matrix Heatmaps

For each model, a **confusion matrix heatmap** is generated.

These visual diagnostics enable:

* Per-class performance assessment
* Identification of systematic misclassifications
* Comparison of class separability across models

This level of analysis is essential for understanding real-world model behavior beyond aggregate accuracy.

---

### Cross‑Model Comparison
The overall comparison plot highlights differences in convergence and accuracy across all models:

![Accuracy Comparison Plot](assets/models_comparison_acc.png)
![Loss Comparison Plot](assets/models_comparison_loss.png)

---

### Model‑Specific Diagnostics

<details>
<summary>CNN from Scratch</summary>

**Loss & Accuracy Curves**
<img src="results/acc_curve_cnn.png" width="400">
<img src="results/loss_curve_cnn.png" width="400">

**Confusion Matrix**
<img src="results/confusion_matrix_heatmap_cnn.png" width="400">

</details>

<details>
<summary>ResNet50</summary>

**Loss & Accuracy Curves**
<img src="results/acc_curve_resnet.png" width="400">
<img src="results/loss_curve_resnet.png" width="400">

**Confusion Matrix**
<img src="results/confusion_matrix_heatmap_resnet.png" width="400">

</details>

<details>
<summary>MobileNetV2</summary>

**Loss & Accuracy Curves**
<img src="results/acc_curve_mobilenet.png" width="400">
<img src="results/loss_curve_mobilenet.png" width="400">

**Confusion Matrix**
<img src="results/confusion_matrix_heatmap_mobilenet.png" width="400">

</details>

<details>
<summary>EfficientNetB0</summary>

**Loss & Accuracy Curves**
<img src="results/acc_curve_efficientnet.png" width="400">
<img src="results/loss_curve_efficientnet.png" width="400">

**Confusion Matrix**
<img src="results/confusion_matrix_heatmap_efficientnet.png" width="400">

</details>

All plots are embedded below for direct inspection. Original high-resolution files are also stored in the [`results/`](results/) directory.

---

## Results Summary

### Quantitative Performance Overview

| Model            | Accuracy | Relative Performance | Key Observations                            |
|------------------|----------|--------------------|-----------------------------------------------|
| CNN from Scratch | 43.3%    | Lowest             | Limited generalization, slower convergence    |
| ResNet50         | 91.2%    | Very High          | Stable training, strong feature extraction    |
| MobileNetV2      | 90.2%    | High               | Optimal balance of accuracy and efficiency    |
| EfficientNetB0   | 92.2%    | Very High          | Consistent accuracy with compact architecture |

> Exact numerical metrics are reported in the notebook to preserve experimental integrity.

### Qualitative Insights

* Transfer learning consistently outperforms training from scratch
* Efficient architectures generalize well on visually similar classes
* Confusion matrices confirm improved class-level discrimination

---

## Key Takeaways

* Transfer learning substantially enhances classification performance
* Model efficiency does not necessarily compromise accuracy
* Visual evaluation is critical for reliable model assessment

---

## Future Directions

* Fine-tuning deeper layers of pretrained networks
* Addressing class imbalance
* Hyperparameter optimization
* Evaluation using newer architectures (EfficientNetV2, ConvNeXt)

---

## Project Structure

```
├── notebook.ipynb   # Complete implementation, experiments, and visualizations
├── README.md       # Project overview and analytical summary
```

---

## Environment & Dependencies
* Python 3.10+  
* TensorFlow 2.x  
* NumPy, Pandas, Matplotlib, Seaborn  
* scikit-learn  
* Jupyter Notebook / JupyterLab  

> Installing via `pip install -r requirements.txt` recommended for reproducibility.

---

## Reproducibility and Transparency

* All metrics and figures are generated directly within the notebook
* Experiments progress sequentially from baseline to advanced models
* Visual evidence is preserved to support all conclusions

---

**Author:** *Arian Jr*  
**Focus Areas:** Computer Vision · Deep Learning · Transfer Learning  
