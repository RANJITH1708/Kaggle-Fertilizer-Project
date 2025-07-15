# Kaggle Competition: Fertilizer Optimization Project

This repository contains the code and analysis for a project aimed at predicting crop yield based on fertilizer application and soil data, inspired by a Kaggle competition.

## Project Overview

The primary goal of this project is to analyze the relationship between different types of fertilizers, soil conditions, and crop yields. By leveraging machine learning, this project seeks to build a model that can provide recommendations for optimizing fertilizer usage, ultimately leading to better agricultural outcomes.

## Key Features

* **Data Exploration:** In-depth exploratory data analysis (EDA) to understand trends and relationships in the dataset.
* **Model Building:** Implementation of various machine learning models to predict crop yield.
* **Ensemble Methods:** Explored complex ensemble strategies to improve final prediction accuracy.

## Technologies Used

* Python
* Jupyter Notebook
* Pandas for data manipulation
* Scikit-learn for machine learning modeling
* Matplotlib / Seaborn for data visualization

## Results

The models were evaluated based on the private score metric from the Kaggle competition. The best-performing model, using an ensemble grandmaster strategy, achieved a final private score of **0.35508**.

### Submission Leaderboard
| Submission File | Public Score | Private Score |
| :--- | :--- | :--- |
| `submission_grandmaster_strategy.csv` | 0.35118 | **0.35508** |
| `submission_optimized_ensemble.csv` | 0.35012 | 0.35408 |
| `submission.csv` | 0.34363 | 0.34487 |
