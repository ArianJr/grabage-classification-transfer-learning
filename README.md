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
| Loss Function       | Sparse Categorical Crossentropy                |
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

---

### Confusion Matrix Heatmaps

For each model, a **confusion matrix heatmap** is generated.

These visual diagnostics enable:

* Per-class performance assessment
* Identification of systematic misclassifications
* Comparison of class separability across models

This level of analysis is essential for understanding real-world model behavior beyond aggregate accuracy.

---

## Results Summary

### Quantitative Performance Overview

| Model            | Accuracy | Relative Performance | Key Observations                                |
|-----------------|----------|--------------------|-----------------------------------------------|
| CNN from Scratch | 72.4%    | Lowest             | Limited generalization, slower convergence    |
| ResNet50         | 91.3%    | High               | Stable training, strong feature extraction    |
| MobileNetV2      | 92.1%    | Very High          | Optimal balance of accuracy and efficiency    |
| EfficientNetB0   | 92.5%    | Very High          | Consistent accuracy with compact architecture |

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
