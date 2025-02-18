# Car Price Prediction using Regression

## Project Overview
This project focuses on building and evaluating regression models to predict car prices using machine learning techniques. The goal is to analyze different regression models, optimize their performance through feature selection and regularization, and select the best-performing model. Both linear and nonlinear regression models will be implemented and compared.

## Dataset
We are using the [Cars Dataset from YallaMotors](https://www.kaggle.com/datasets/ahmedwaelnasef/cars-dataset/data), which contains approximately 6,750 rows and 9 columns. The dataset includes various features such as:
- Car make and model
- Year of manufacture
- Mileage
- Engine size
- Price (target variable)

### Data Preprocessing Steps
- Handle missing values.
- Encode categorical features.
- Normalize/standardize numerical features if necessary.
- Convert car prices to a uniform currency (e.g., USD) to maintain consistency.
- Split the dataset into training (60%), validation (20%), and testing (20%) sets.

## Regression Models Implemented
### Linear Models
- **Linear Regression**
- **LASSO Regression** (L1 Regularization)
- **Ridge Regression** (L2 Regularization)
- **Closed-form solution for Linear Regression** (Implemented manually without external ML libraries)
- **Gradient Descent for Linear Regression** (Comparison with closed-form solution)

### Nonlinear Models
- **Polynomial Regression** (Varying degrees from 2 to 10)
- **Radial Basis Function (RBF) Regression** (Using a Gaussian kernel)

## Model Selection & Evaluation
### Performance Metrics
To compare the models, we will use:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

The best model will be selected based on the lowest MSE or highest R² on the validation set.

### Feature Selection
We will implement **Forward Selection**, where:
1. The model starts with an empty feature set.
2. Features are added iteratively based on the best performance improvement.
3. The process stops when adding features no longer improves performance.

### Regularization Techniques
To prevent overfitting and improve model generalization:
- **LASSO Regression** will be used to reduce unnecessary features.
- **Ridge Regression** will be used to penalize large coefficients.
- **Grid Search** will be used to find the optimal regularization parameter (λ).

### Hyperparameter Tuning
A **Grid Search** approach will be applied to optimize model parameters, such as:
- λ values for LASSO and Ridge Regression.
- Polynomial degrees for Polynomial Regression.

## Final Model Evaluation
After selecting the best model using the validation set, it will be evaluated on the test set to measure its generalization performance. The final report will include:
- Dataset description and preprocessing steps.
- Comparison of different regression models.
- Feature selection process and results.
- Regularization outcomes and selected λ values.
- Hyperparameter tuning details.
- Final model performance on the test set.
- Visualizations of feature importance, error distribution, and model predictions.

## Optional Extension
For further exploration, we may:
- Identify another relevant target variable from the dataset and build a regression model to predict it.

## Authors
This project is completed by:
- Ali Shaikh Qasem.
- Abdelrahamn Jaber.

