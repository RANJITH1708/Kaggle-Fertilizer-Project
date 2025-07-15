# Predicting Optimal Fertilizers: A Kaggle Competition Deep Dive

This repository contains the complete, end-to-end machine learning pipeline developed for the **Kaggle Playground Series - Season 5, Episode 6: Predicting Optimal Fertilizers** competition.

The final model, a weighted ensemble of highly-tuned XGBoost and LightGBM models, achieved a **Public Leaderboard MAP@3 Score of 0.34663**.

## 1. Project Objective

The goal of this project was to develop a model that accurately predicts the optimal fertilizer type for a given crop based on various soil and environmental conditions. This is a multi-class classification problem with the performance evaluated on the Mean Average Precision @3 (MAP@3) metric.

This notebook documents the full workflow, including:
* In-depth Exploratory Data Analysis (EDA)
* Multiple Feature Engineering Strategies
* GPU-Accelerated Hyperparameter Tuning with Optuna
* Model Selection and Advanced Ensembling

## 2. Data Sources

To build a robust model, this project utilized two key data sources:
1.  **Competition Data:** The primary `train.csv` and `test.csv` files provided by the competition.
2.  **External Data:** The [original "Fertilizer Prediction" dataset](https://www.kaggle.com/datasets/emekaorji/fertilizer-prediction), which was used to augment the training set, giving the models more data to learn from. This was a key strategy identified from top-performing public notebooks.

## 3. Exploratory Data Analysis (EDA)

Initial analysis revealed a clean dataset with no missing values and well-balanced classes, providing a strong foundation for modeling.

#### Numerical Feature Distributions
The numerical features were found to be well-behaved, without extreme outliers or heavy skew, making them suitable for direct use in tree-based models.
*(Here you would add an image you saved, like your histogram plot)*
`![Numerical Distributions](images/your_histogram_plot.png)`

#### Target Variable Distribution
The 7 fertilizer classes in the combined dataset were found to be relatively balanced, which is ideal for a classification task and reduces the need for complex sampling strategies, although class weights were used to further enhance performance.
*(Here you would add your fertilizer count plot)*
`![Fertilizer Counts](images/your_fertilizer_countplot.png)`

## 4. Methodology and Pipeline

The final solution was the result of systematic experimentation. The pipeline that produced the best score followed these steps:

**1. Data Combination & Preprocessing:**
* The official training data was combined with the external dataset to create a larger training set of over 850,000 samples.
* Categorical features (`Soil Type`, `Crop Type`) were encoded using integer codes (`.cat.codes`), which proved more effective for XGBoost than one-hot encoding.
* The target variable (`Fertilizer Name`) was label encoded.

**2. Champion Model Tuning:**
* **Optuna**, a hyperparameter optimization framework, was used to run over 100 trials for each model (LightGBM, XGBoost, CatBoost).
* All tuning was **accelerated using a local NVIDIA GPU (CUDA)** to make these extensive searches feasible.
* **XGBoost** was identified as the strongest single model (CV Score: `~0.344`), followed by LightGBM.

**3. Final Ensemble Model:**
* The final submission was created by training the two best models (XGBoost and LightGBM) on the full, combined dataset using their champion hyperparameters found by Optuna.
* The predictions from these two models were blended using a **weighted average** (`0.55 * XGBoost + 0.45 * LightGBM`) to produce the final probabilities. This method proved more effective than stacking with a neural network or a simple 3-model average.

## 5. Results

* **Best Single Model (Tuned XGBoost) CV Score:** `0.34358` MAP@3
* **Final Ensemble Public Leaderboard Score:** `0.34663` MAP@3
* **Final Rank:** 232

## 6. How to Reproduce

The `Final Notebook.ipynb` file in this repository contains the complete, commented code to reproduce this result from start to finish. To run it:
1.  Set up a Python environment (Python 3.11 recommended).
2.  Install the necessary libraries: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`. Ensure they are GPU-enabled versions if you have a compatible NVIDIA GPU and CUDA installed.
3.  Download the two required datasets from Kaggle and place them in a `/data` subfolder.
4.  Run the notebook cells sequentially.
