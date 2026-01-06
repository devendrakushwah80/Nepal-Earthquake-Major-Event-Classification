# 🌍 Nepal Earthquake Major Event Classification  
*(Using Nepal Seismic Data)*

## 📌 Project Overview

Earthquakes vary significantly in their **severity and impact**.  
Early identification of **major earthquakes** can help in better risk assessment and preparedness.

In this project, the goal is to **predict whether an earthquake is a major event or not** using spatial, temporal, and historical seismic features from **Nepal seismicity data**.

The problem is formulated as a **binary classification task**, where machine learning models learn patterns associated with major seismic events.

---

## 📊 Dataset Description

The dataset consists of **historical earthquake records from Nepal** with the following features:

- **Spatial features:**  
  Latitude, Longitude, Location Offsets  

- **Seismic features:**  
  Magnitude, Rolling earthquake counts, Rolling mean magnitude  

- **Temporal features:**  
  Year, cyclical month and hour encodings  

- **Event history features:**  
  Days since last major earthquake  

- **Metadata:**  
  Data source  

---

## 🎯 Target Variable

### `is_major` (Binary Classification Target)

- `0` → Non-major earthquake  
- `1` → Major earthquake  

> ⚠️ To avoid **data leakage**, features directly derived from the target definition were carefully reviewed.

---

## 🔍 Exploratory Data Analysis (EDA)

Key analyses performed:

- Dataset shape, data types, and memory usage
- Class distribution analysis of `is_major`
- Missing value and duplicate checks
- Feature-wise distributions
- Relationship between magnitude, time, and major earthquake occurrence
- Temporal patterns of major events

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

- Dropped non-informative or high-cardinality columns:
  - `place`
  - raw datetime (`dt`)
- Removed depth-related columns to keep the task focused on classification:
  - `depth`
  - `depth_log`
- Label Encoding for categorical feature (`source`)
- Train–Test split using **80/20 ratio**
- Feature scaling using **StandardScaler**
- Ensured consistent preprocessing across train and test sets

---

## 🤖 Machine Learning Models

Multiple **binary classification models** were implemented and compared:

### 1️⃣ Logistic Regression
- Linear baseline classifier
- Interpretable coefficients
- Probability-based predictions
- Evaluated using ROC–AUC

### 2️⃣ K-Nearest Neighbors (KNN)
- Tested across multiple values of **K**
- Best K selected based on validation performance
- Demonstrates importance of feature scaling

### 3️⃣ Support Vector Machine (SVM)
- Kernel-based classifier (RBF kernel)
- Handles non-linear decision boundaries
- Probability estimates enabled for ROC–AUC comparison

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision, Recall, and F1-score
- ROC–AUC Score
- Confusion Matrix

> Since **major earthquakes are relatively rare**, ROC–AUC and Recall were emphasized over accuracy.

---

## 🏆 Model Comparison Summary

| Model                | Accuracy | ROC–AUC |
|---------------------|----------|---------|
| Logistic Regression | ✓        | ✓       |
| KNN                 | ✓        | ✓       |
| SVM                 | ✓        | ✓       |

*(Exact scores may vary depending on random state and preprocessing.)*

---

## 🧠 Key Learnings

- Major earthquake prediction is a **challenging and imbalanced classification problem**
- Temporal and historical seismic features add strong predictive value
- Feature scaling is crucial for KNN and SVM
- ROC–AUC is a more reliable metric than accuracy for rare events
- Avoiding **data leakage** is critical for building realistic models

---

## 🧾 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
