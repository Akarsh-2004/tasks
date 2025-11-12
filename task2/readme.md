# Task 2: Building and Improving a Baseline Machine Learning Model

## Overview

This task demonstrates a fundamental understanding of building and improving a baseline machine learning model. The goal is to showcase skills in data preprocessing, model building, evaluation, and improvement techniques.

**Dataset:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
**Task:** Multi-class classification of iris species  
**Programming Language:** Python  
**Libraries Used:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `numpy`

## Project Structure

```

├── model.ipynb               # Jupyter Notebook containing full ML workflow
├── python_executable.py      # Python script version of notebook
└── README.md                 # This file
```

---

## Methodology

### 1. Baseline Model

- **Model Used:** Support Vector Classifier (SVC) with default parameters  
- **Evaluation Metric:** Accuracy and Classification Report  
- **Result:**  
  - Test Accuracy: **96.7%**  
  - Baseline SVM performs well on the Iris dataset without preprocessing

### 2. Improvement 1: Feature Scaling

- **Why:** SVM is sensitive to feature magnitude  
- **Method:** StandardScaler (zero mean, unit variance)  
- **Result:** Accuracy remained **96.7%**, indicating baseline SVM was robust for this dataset

### 3. Improvement 2: Hyperparameter Tuning

- **Method:** GridSearchCV to tune `C`, `kernel`, and `gamma` parameters  
- **Best Parameters:**  
  ```
  C=0.1, kernel=linear, gamma=scale
  ```
- **Result:** Test accuracy slightly decreased to **93.3%**  
- **Observation:** While tuning can optimize cross-validation performance, it may not always improve small test set accuracy. Provides insight into the effect of hyperparameters on SVM

---

## Comparison of Models

| Model                          | Test Accuracy |
|--------------------------------|---------------|
| Baseline SVM                   | 0.967         |
| SVM + Scaling                  | 0.967         |
| SVM + Scaling + GridSearch     | 0.933         |

---

## Key Takeaways

- Baseline SVM already performs well on simple datasets like Iris
- Feature scaling is a good practice for SVM but may not always change performance for clean datasets
- Hyperparameter tuning is useful to understand parameter effects but does not always guarantee higher test accuracy
- Evaluation using accuracy and classification reports ensures clear, quantitative comparison between models

---

## How to Run

1. **Install required libraries:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   ```

2. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook model.ipynb
   ```

   **OR**

3. **Execute the Python script:**
   ```bash
   python python_executable.py
   ```

---
