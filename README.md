This project reimplements and evaluates six machine learning algorithms for predicting student academic performance, based on the systematic review by Alsariera et al. (2022). Using the UCI Student Performance dataset (Math and Portuguese courses), models are compared across accuracy, F1-score, precision, recall, and cross-validation stability. The project is structured according to the CRISP-DM methodology, covering data exploration, preprocessing, modeling, evaluation, and deployment considerations.

# Student Performance Prediction — CRISP-DM Final Project

## Overview
Reimplementation of:
> **"Assessment and Evaluation of Different Machine Learning Algorithms for Predicting Student Performance"**  
> Alsariera et al., *Computational Intelligence and Neuroscience*, 2022  
> DOI: https://doi.org/10.1155/2022/4151487

This project reimplements and evaluates 6 ML algorithms from the paper (+ 2 bonus models) on the UCI Student Performance Dataset, organized using the **CRISP-DM** methodology.

---

## Project Structure

```
student_performance_project/
├── student_performance_crisp_dm.ipynb   # Main notebook (source)
├── student_performance_executed.ipynb   # Executed notebook with outputs
├── student-mat.csv                      # Math course dataset
├── student-por.csv                      # Portuguese course dataset
├── student.txt                          # Dataset attribute description
├── best_student_model.pkl               # Saved best model (joblib)
├── scaler_math.pkl                      # Saved StandardScaler (math)
├── README.md                            # This file
└── figures/
    ├── fig_grade_distribution.png
    ├── fig_categorical_distributions.png
    ├── fig_correlation_heatmap.png
    ├── fig_feature_relationships.png
    ├── fig_grade_progression.png
    ├── fig_class_balance.png
    ├── fig_accuracy_comparison.png
    ├── fig_confusion_matrices.png
    ├── fig_feature_importance.png
    ├── fig_radar_chart.png
    ├── fig_cv_boxplot.png
    └── fig_summary_dashboard.png
```

---

## How to Run

### Requirements
```bash
pip install jupyter scikit-learn pandas numpy matplotlib seaborn joblib
```

### Run the Notebook
```bash
jupyter notebook student_performance_crisp_dm.ipynb
```
Or open the pre-executed version:
```bash
jupyter notebook student_performance_executed.ipynb
```

### Use Saved Model for Inference
```python
import joblib, pandas as pd

model  = joblib.load('best_student_model.pkl')
scaler = joblib.load('scaler_math.pkl')

# Prepare your student features (same columns as training data)
X_new_scaled = scaler.transform(X_new)
prediction   = model.predict(X_new_scaled)        # 1=Pass, 0=Fail
probability  = model.predict_proba(X_new_scaled)[:,1]  # Pass probability
```

---

## CRISP-DM Phases

| Phase | Description |
|-------|-------------|
| **1. Business Understanding** | Predict student pass/fail to enable early intervention |
| **2. Data Understanding** | EDA on 395 (Math) + 649 (Portuguese) students, 33 features each |
| **3. Data Preparation** | Label encoding, one-hot encoding, binary target creation, train/test split, StandardScaler |
| **4. Modeling** | DT, ANN, SVM, KNN, NB, LogReg + RF, GradBoost (bonus) |
| **5. Evaluation** | Accuracy, F1, Precision, Recall, 5-fold CV, comparison with paper |
| **6. Deployment** | Model serialization, inference API sketch, ethical considerations |

---

## Models Implemented

| Model | In Paper | Paper Avg Acc |
|-------|----------|--------------|
| Decision Tree | ✓ | 85.0% |
| ANN (MLP) | ✓ | 85.9% |
| SVM | ✓ | 83.4% |
| KNN | ✓ | 80.7% |
| Naive Bayes | ✓ | 83.0% |
| Logistic Regression | ✓ | 55.5% |
| Random Forest | 🎁 Bonus | — |
| Gradient Boosting | 🎁 Bonus | — |

---

## Dataset
**UCI Student Performance Dataset** (Cortez & Silva, 2008)  
Source: https://archive.ics.uci.edu/dataset/320/student+performance

Features include: demographic (sex, age, address), family (parents' education/job), academic (studytime, failures, absences, G1, G2), and social attributes.

**Target**: Binary classification — Pass (G3 ≥ 10) / Fail (G3 < 10)

---

## Key Findings
- Prior grades (G2, G1) are the most predictive features — consistent with paper
- Bonus models (Random Forest, Gradient Boosting) outperform all paper models
- Removing G1/G2 reveals which demographic/behavioral factors matter for early prediction
- ANN and DT are the strongest paper models — aligns with paper's conclusion

---

## Reference
Alsariera, Y. A., Baashar, Y., Alkawsi, G., Mustafa, A., Alkahtani, A. A., & Ali, N. (2022).
Assessment and Evaluation of Different Machine Learning Algorithms for Predicting Student Performance.
*Computational Intelligence and Neuroscience*, 2022, Article ID 4151487.
https://doi.org/10.1155/2022/4151487
