#  Disaster Damage Assessment using xBD Dataset

##  Overview

This project performs a comprehensive **analysis and modeling of post-disaster building damage** using satellite imagery from the **xBD (xView2)** dataset.

The primary goal is to evaluate whether standard **damage severity labels** (`no-damage`, `minor-damage`, `major-damage`, `destroyed`) are sufficient to represent visual variability in post-disaster imagery.

The project combines:

* **Multivariate statistical analysis**
* **Deep feature extraction (ResNet-50)**
* **Deep learning models (CNNs & Transformers)**
* **Tile-level and building-level experiments**
* **Dual-stream pre/post-disaster modeling**

---

##  Key Contributions

* Quantitative evaluation of **label coherence** in feature space
* Comparison of **severity vs disaster type** as organizing factors
* Extensive **model benchmarking** across architectures
* Identification of **minor-damage as the most challenging class**
* Exploration of **building-level vs tile-level learning**
* Implementation of a **dual-stream change detection model**

---

##  Dataset

* **Source:** xBD (xView2)
* **Format:** GeoTIFF images + JSON annotations
* **Split used:** `hold`

### Dataset Statistics

| Level          | Samples |
| -------------- | ------- |
| Tile-level     | 751     |
| Building-level | 52,537  |

### Class Distribution (Building-level)

* No-damage: 37,505
* Minor-damage: 6,302
* Major-damage: 4,531
* Destroyed: 4,199

⚠️ Dataset is highly imbalanced.

---

## ⚙️ Tech Stack

* **PyTorch / TorchVision**
* **PySpark** (distributed preprocessing)
* **Rasterio** (GeoTIFF handling)
* **scikit-learn**
* **OpenCV / NumPy / Pandas**
* **Matplotlib / Seaborn**

Environment:

* Google Colab (T4 GPU)

---

##  Project Pipeline

### 1. Data Processing

* GeoTIFF loading and normalization
* NoData masking
* Cropping to valid regions
* Image-label matching via base IDs

### 2. Feature Extraction

* Pretrained **ResNet-50**
* 2048-dimensional embeddings
* Distributed extraction using PySpark

### 3. Label Processing

* JSON parsing (WKT polygons)
* Building-level → tile-level aggregation
* Disaster metadata extraction

### 4. Statistical Analysis

* PCA (dimensionality reduction)
* Pairwise distance analysis
* Silhouette score
* PERMANOVA (variance attribution)

### 5. Predictive Probing

* Logistic Regression
* Linear SVM
* 5-fold stratified cross-validation

### 6. Deep Learning Experiments

* Tile-level classification
* Building-level classification
* Binary and 4-class setups

### 7. Dual-Stream Modeling

* Pre + Post disaster images
* Siamese MobileNetV3-Large
* Attention-based feature fusion

---

##  Key Findings

### 🔹 Feature Space Analysis

* High intrinsic dimensionality
* No dominant visual structure

### 🔹 Label Coherence

* Intra-class ≈ Inter-class distance
* Silhouette ≈ 0 (very weak clustering)

### 🔹 Disaster vs Severity

* Disaster type explains more variance than severity
* Higher separability and predictability

### 🔹 Predictive Performance

* Disaster classification: ~91% accuracy
* Severity classification: ~56% accuracy

### 🔹 Deep Learning Results

| Model                      | Level    | Accuracy  | Macro-F1  |
| -------------------------- | -------- | --------- | --------- |
| ResNet-50                  | Tile     | 0.623     | 0.543     |
| EfficientNet-B0            | Tile     | 0.596     | 0.492     |
| MobileNetV3-Small          | Tile     | 0.596     | 0.543     |
| **ViT-B/16**               | Tile     | **0.682** | **0.546** |
| Pre+Post MobileNetV3-Large | Tile     | 0.609     | 0.534     |
| EfficientNet-B0 (4-class)  | Building | 0.681     | 0.552     |
| EfficientNet-B0 (binary)   | Building | 0.805     | 0.666     |

---

##  Important Insights

* Severity labels are **not visually coherent**
* Disaster type is a **stronger organizing factor**
* Minor-damage is:

  * Highly ambiguous
  * Poorly learned across all models
* Building-level modeling improves performance but **does not fully solve the problem**
* Dual-stream temporal modeling provides **limited gains**

---

##  Experiments

### Tile-Level Models

* ResNet-50
* EfficientNet-B0
* MobileNetV3-Small
* ViT-B/16
* CLIP (frozen)

### Building-Level Models

* EfficientNet-B0 (frozen backbone)
* MobileNetV3-Small
* Binary (minor vs rest)

### Advanced Model

* Dual-stream Siamese MobileNetV3-Large
* Attention-based fusion
* Change-aware representation

---

## 📁 Project Structure

```
.
├── DataProcessing_and_ModelRunning.ipynb
└── README.md
```

---

## ▶️ How to Run

1. Mount Google Drive (Colab)
2. Extract xBD dataset
3. Run notebook sequentially:

   * Data preprocessing
   * Feature extraction
   * Analysis
   * Model training

---

## ⚠️ Limitations

* Severe class imbalance
* Tile-level labels are coarse (max severity)
* Bounding-box approximation for building crops
* Possible data leakage at building level (same tile split)

---

##  Future Work

* Multi-factor labels (severity + disaster type)
* Continuous damage representation
* Better change detection models
* Graph-based building relationships
* Domain adaptation across disasters

---

##  Conclusion

This work shows that:

> Standard damage severity labels are **insufficient to represent the true visual variability** in post-disaster satellite imagery.

More expressive, multi-dimensional representations are needed for robust disaster damage assessment.

---

## 🙌 Acknowledgements

* xBD / xView2 dataset creators
* PyTorch & open-source community


