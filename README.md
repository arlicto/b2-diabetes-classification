# Diabetes Classification with PyTorch

## Overview

This project explores binary diabetes classification using artificial neural networks implemented with **PyTorch**. It uses the Pima Indians Diabetes Dataset to compare three classification approaches and evaluate their performance on unseen data.

The project demonstrates fundamental machine learning concepts, including data preprocessing, neural network architectures, model training, evaluation metrics, and error analysis.

## Objectives

- Build a majority-class baseline for binary classification.
- Implement a single-neuron model using PyTorch.
- Build a multilayer perceptron (MLP).
- Compare model performance using standard classification metrics.
- Understand loss functions, gradients, and backpropagation.
- Identify the limitations of machine learning models in healthcare applications.

## Models

Three models are compared:

| Model | Description |
|---|---|
| **M0 — Majority Baseline** | Predicts the most frequent class in the training dataset. |
| **M1 — Single Neuron** | Uses a single linear layer with 8 input features and 1 output. |
| **M2 — Multilayer Perceptron** | Uses an 8-neuron hidden layer with ReLU activation and a single output neuron. |

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains 768 observations and 8 input features.

The features are:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is `Outcome`:

- `0` — No diabetes
- `1` — Diabetes

Dataset source: [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- Jupyter Notebook

## Methodology

### 1. Data Preprocessing

- Inspect the dataset and identify missing or implausible values.
- Replace zero values with missing values in selected medical measurements.
- Split the data into training, validation, and test sets using stratified sampling.
- Impute missing values using training-set medians.
- Standardize input features using statistics calculated from the training set.
- Avoid data leakage by fitting preprocessing steps only on the training data.

### 2. Neural Network Training

The neural networks are trained using:

- **Loss function:** `BCE
