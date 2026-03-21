# Predicting Alzheimer's Disease: A Comparative Study of Multilayer Perceptron & Tab Transformer Models

## Project Overview

This project investigates the use of deep learning models to predict Alzheimer’s disease diagnosis from structured clinical and lifestyle data. Alzheimer’s disease is a progressive neurodegenerative condition that affects cognitive functioning and daily living, making early risk identification an important objective in healthcare analytics.

We formulate the task as a binary classification problem using heterogeneous tabular features, including demographic information, medical history, clinical measurements, cognitive assessments, and reported symptoms. To address challenges such as feature scaling, moderate class imbalance, and nonlinear feature interactions, we design and implement two neural architectures in PyTorch: a feedforward multilayer perceptron (MLP) baseline and a transformer-based TabTransformer that leverages self-attention to model relationships between features.

Through feature engineering, categorical embeddings, weighted loss functions, and early stopping strategies, the project evaluates the effectiveness of modern deep learning approaches for tabular healthcare prediction using metrics such as accuracy, F1-score, and ROC-AUC.

## Setup Instructions/Reproducing Results

Step 1 - Download dataset: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

Step 2 - Once the dataset is downloaded, place it in a folder titled 'data'

Step 3 - Download 'Alzheimer's Disease Prediction - Complete Notebook.ipynb' file (Found in this repository)

Step 4 - Make sure dependencies (listed below) are installed

Step 5 - Launch Jupyter Notebook & run the notebook

## Required Dependencies

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

## How to Preprocess the Data

Data preprocessing is handled automatically within the notebook through the `load_and_preprocess_data()` function.

This pipeline performs the following steps:

* Loads the Alzheimer’s disease dataset and removes non-informative columns
* Engineers additional clinical features (e.g., blood pressure interaction and cholesterol ratios)
* Splits the data into stratified training, validation, and test sets
* Standardizes numerical features using `StandardScaler`
* Encodes categorical features using integer representations for embedding layers
* Computes class weights to address moderate class imbalance
* Converts the processed data into PyTorch tensors and creates `DataLoader` objects for model training

To preprocess the data, simply run the preprocessing cells before starting model training.

## How to Train the Model

Training is performed directly within the notebook file.

After setup instructions, run the notebook cells sequentially until reaching the **Main Training & Evaluation** section. This step loads and preprocesses the dataset, creates data loaders, initializes the models (MLP and TabTransformer), and begins training.

Both models are trained using weighted binary cross-entropy loss, Adam optimization, learning rate scheduling, and early stopping based on validation performance. The best model weights are automatically saved during training.

## How to Evaluate the Model

Evaluation is also handled directly in the notebook after each model finishes training.

The evaluate_model(...) function is used on the held-out test set to compute:

- Accuracy
- F1-Score
- ROC-AUC
- Precision
- Recall
- Specificity
- Confusion Matrix

After training each model, the notebook:

prints the full set of test metrics
plots training and validation loss curves
displays a confusion matrix heatmap

To evaluate models, simply run the Main Training & Evaluation section after training. The notebook will automatically evaluate both the MLP and TabTransformer on the test set and display the corresponding performance summaries and plots.

## Expected Outputs

Running the notebook sequentially produces:

* Training logs showing epoch-wise training and validation loss
* Model performance metrics on the test set, including Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Specificity
* Training and validation loss curves for both the MLP and TabTransformer models
* Confusion matrix visualizations for each model
* Exploratory Data Analysis (EDA) plots (diagnosis distribution, feature boxplots, correlation heatmaps, and feature importance rankings)

These outputs allow for performance comparison between model architectures and help assess how well each model generalizes to unseen data.

