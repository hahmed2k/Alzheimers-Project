# Predicting Alzheimer's Disease: A Comparative Study of Multilayer Perceptron & Tab Transformer Models

## Project Overview

This project explores the use of deep learning models for predicting Alzheimer’s disease diagnosis using structured clinical and lifestyle data. Alzheimer’s disease is a progressive neurodegenerative condition that affects memory, cognitive functioning, and daily living abilities, making early identification an important objective in healthcare analytics. Leveraging machine learning techniques on patient health data can help uncover patterns associated with disease risk and support decision-making in clinical settings.

The task is formulated as a binary classification problem, where the goal is to predict whether a patient has been diagnosed with Alzheimer’s disease based on demographic information, lifestyle factors, medical history, clinical measurements, cognitive assessments, and reported symptoms. The dataset consists of heterogeneous tabular features, including continuous, categorical, and binary variables, presenting challenges such as feature scaling, moderate class imbalance, and complex nonlinear relationships.

To address this problem, we design and implement two deep learning architectures using PyTorch: a multilayer perceptron (MLP) baseline model and a transformer-based TabTransformer model that leverages self-attention to capture interactions between features. Additional components of the modeling pipeline include feature engineering, categorical embeddings, weighted loss functions to handle class imbalance, and early stopping strategies to improve generalization.

Through comparative experimentation and performance evaluation using metrics such as accuracy, F1-score, and ROC-AUC, this project aims to assess the effectiveness of modern neural architectures for tabular healthcare prediction tasks and to provide insights into the predictive importance of cognitive, behavioral, and physiological features associated with Alzheimer’s disease.


## Setup Instructions

Step 1 - Download dataset: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

Step 2 - Once the dataset is downloaded, place it in a folder titled 'data'

Step 3 - Download 'Alzheimer's Disease Prediction - Complete Notebook.ipynb' file

Step 4 - Make sure dependencies listed below are installed

Step 5 - Launch Jupyter Notebook & run the notebook

## Required dependencies

Project requires Python versions 3.8+ and the following libraries:
- Pandas: Used for data loading, cleaning, and structured manipulation of Alzheimer's dataset
- NumPy: Supports large, multi-dimensional array processing and mathematical functions
- Scikit-Learn: Used primarily for data preprocessing (stratified sampling/splitting, scaling, etc)
- PyTorch: The core deep learning framework used to build, train, and optimize MLP and TabTransformer models
- Matplotlib & Seaborn: Used for EDA, and related visualizations

Dependencies can be installed by running the following in terminal:

```
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

## How to train the model

## How to evaluate the model

## Expected Outputs
