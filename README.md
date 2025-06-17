#  Diabetes Prediction using Machine Learning and Deep Learning

A data-driven project built to predict the likelihood of diabetes in individuals using advanced machine learning and deep learning models. This project was developed as part of the Data Science course .

## Project Overview

The rise in diabetes due to sedentary lifestyles and unhealthy eating habits calls for efficient, automated diagnostic tools. This project aims to build a robust classification model using the Indian PIMA dataset to predict whether a person is diabetic or not.

This work compares four supervised learning algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- Gradient Boosting
- Feedforward Neural Network (FNN)

Among these, **Gradient Boosting** achieved the best performance with **84.5% accuracy** and an **AUC score of 0.8999**.

##  Dataset

I used the **PIMA Indian Diabetes Dataset**, which consists of **768 female patients** and the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (1 for diabetic, 0 for non-diabetic)

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras
- Seaborn & Matplotlib
- SMOTE (Synthetic Minority Oversampling Technique)

##  Methodology

### 1. Data Preprocessing
- Removal of duplicates and imputation of missing/zero values
- Data normalization
- Class balancing using **SMOTE**

### 2. Data Visualization
- Histograms for feature distribution
- Heatmap to identify correlations between variables

![EDA Histogram](figures/Plot1.png)  
![EDA Histogram](figures/plot2.png)  
![Correlation Heatmap](figures/correlationmap.png)

### 3. Model Training & Evaluation

| Model                 | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | 77.5%    | 0.86    |
| Support Vector Machine | 78.5%    | 0.8659  |
| Gradient Boosting    | **84.5%** | **0.8999** |
| Feedforward Neural Network | 79.5%    | ~0.88    |

Each model was trained using a balanced dataset and evaluated using:
- Accuracy
- ROC-AUC
- Classification Report

![Output](figures/Output.png)

##  Neural Network Architecture

- Input Layer: 8 Features
- Dense Layer 1: 256 Neurons, ReLU, BatchNorm, Dropout (30%)
- Dense Layer 2: 128 Neurons, L2 Regularization, BatchNorm, Dropout (30%)
- Dense Layer 3: 64 Neurons, ReLU, Dropout (20%)
- Output Layer: 1 Neuron, Sigmoid Activation

## 📈 Future Work

- Improving model efficiency and robustness
- Deploying an interactive web application for real-time prediction
- Providing personalized insights and recommendations


## Files in this Repository

| File                 | Description                                 |
|----------------------|---------------------------------------------|
| `diabetes_prediction.py` | Python code for training & evaluating models |
| `diabetes.csv`           | Preprocessed PIMA diabetes dataset         |
| `README.md`              | Project documentation (this file)         |


> **Note:** This project was developed for academic purposes only and is not intended for medical diagnosis or treatment.
