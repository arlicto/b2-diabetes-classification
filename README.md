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

- **Loss function:** `BCEWithLogitsLoss`
- **Optimizer:** Adam
- **Activation function:** ReLU in the MLP
- **Random seed:** 42

Training and validation losses are plotted to observe model learning and help identify potential overfitting.

### 3. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Confusion matrices are generated for the neural network models to examine false positives and false negatives.

## Project Structure

```text
diabetes-classification-pytorch/
├── B2_Diabetes_PyTorch_Group07.ipynb
├── README.md
└── diabetes.csv                 # Download separately; not committed
```

The notebook filename above is the intended submission filename. Adjust it if your actual file uses a different name.

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Download `diabetes.csv` from the dataset source.
3. Upload the CSV file to the Colab session.
4. Run the notebook cells in order.
5. Review the training plots, evaluation metrics, and confusion matrices.

### Option 2: Local Environment

Install the required packages:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn jupyter
```

Place `diabetes.csv` in the notebook's working directory, then launch Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## Results

The notebook compares the three models using accuracy, precision, recall, and F1-score on the test set.

**Add your final results here after running the notebook.** Include the metric values and briefly explain which model performed best on the test set. Do not report estimated or unverified results.

## Key Concepts Demonstrated

- Binary classification
- Artificial neurons and multilayer perceptrons
- Forward propagation and backpropagation
- Trainable weights and biases
- Gradient computation using `.backward()`
- Binary cross-entropy loss
- Training, validation, and test splits
- Data leakage prevention
- Classification metrics and confusion matrices
- False-positive and false-negative analysis

## Limitations

This project is an educational machine learning exercise, not a medical diagnostic tool.

The dataset is relatively small and represents a specific population: women of Pima Indian heritage aged 21 and older. The results may not generalize to other populations or clinical settings.

Model performance also depends on preprocessing choices, training settings, and the classification threshold.

**The models must not be used to diagnose diabetes or make medical decisions.**

## Academic Context

This project was developed as part of a **B2-level academic project** focused on diabetes classification using artificial neurons and PyTorch.

**Group:** 07

## License

This repository is intended for educational purposes. If you add a license, ensure it is compatible with the dataset's terms and any code or materials included in the repository.
