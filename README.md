# Melbourne-Housing-Price-Prediction

## Problem Statement:

In this project, I tried to develop a model which can accurately estimate and predict the price of the residential houses in Melbourne, Australia. The real estate business is of a dynamic nature and a vital aspect of any economy. Home buyers need to make an informed decision before buying any property based on the neighborhood and market reach. Accurately predicting the price of the property is crucial for all the stakeholders involved, namely buyers and sellers. In recent times, the real estate industry increasingly relies on data-driven insights to make a decision. As a result, the need for robust predictive estimation model has increased to harness the abundant data available to generate precise predictions. This dataset provided an opportunity to leverage Machine Learning techniques to build a predictive model that can estimate property price considering various factors. With this project, I aim to discover patterns and relationships within the data, that is, to identify which features strongly influence the house prices, which suburbs would be the best to buy, and which area is the expensive. This also helps the real estate agents to provide recommendations to their clients. The project aims to contribute to the domain by improving accuracy, offering data-backed insights, and ultimately ease stakeholders to make informed decisions in the real estate market.

## Dataset Description:

- This dataset has been taken from the Kaggle website, which is a snapshot of Melbourne Housing Dataset.
- The dataset comprises of 21 features, which includes features like Address, Suburb, Types of Real Estate, Rooms, Real Estate Agent, Price, Date of Sale, Distance from CBD, etc.
- The dataset contains 13580 rows before performing any cleaning or preprocessing.

Dataset Link: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot/data


## Data Preprocessing:


## Machine Learning Algorithms Implementation:

- I applied the following regression algorithms to predict the housing price per unit area based on the input features.

  ### 1. Linear Regression:
  - Linear Regression is a simple algorithm which assumes a linear relationship between the features and the output variable. 
  - It is the simplest model to perform regression task. Hence, I started with this algorithm to create baseline performance.

  ### 2. Bayesian Ridge Regression:
  - Bayesian Ridge regression algorithm is like Ridge regression which also includes a regularization term. The only difference in this algorithm is that it automatically adapts the regularization strength based on the complexity of the data. It provides a probabilistic framework that can estimate the uncertainty associated with the model predictions.
  - The dataset encompasses complex relationship amongst different features. So, Bayesian ridge regression seemed to be a good choice.
  
  ### 3. K-Nearest Neighbors Regressor:
  - K-Nearest Neighbors regression is a non-parametric, instance-based learning algorithm that predicts the target variable based on the majority vote of its k-nearest neighbors. 
  - It is a simple algorithm, less sensitive to outliers and suitable for datasets with non-linear relationships. 
  
  ### 4. Elastic Net:
  - Elastic Net regression is a linear regression model that combines both L1 and L2 regularization. 
  - It works fairly well with datasets having correlated features. The features used in our dataset have varying levels of correlation, as was found out by performing EDA. 
  
  ### 5. Neural Network:
  - Neural Network regression is an algorithm where Artificial Neural Network is used to model and predict continuous numerical values. It uses neurons to simulate the working of a human brain.
  - They can automatically learn complex patterns from data and can handle relationships between input variables and the target variables which are non-linear.
  
  ### 6. Support Vector Regression:
  - Support Vector Regression is an ML algorithm, belonging to the Support Vector Machine (SVM) class. Unlike SVM, SVR has been modified for regression task, where we try to find optimal hyperplane (with maximum margin) in higher dimension to represent the relationship between the attributes and the output variable.
  - SVR works well when the features have complex non-linear relationship amongst each other. This in fact, suits well with the nature of the dataset and was further validated by the comparatively underwhelming performances by traditional linear regression models.

