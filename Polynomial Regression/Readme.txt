Predictive Analysis of Blood Pressure with Polynomial, Multiple Linear, and Ridge Regression

Purpose:
This project investigates the relationships between various patient factors (e.g., cholesterol, weight, smoking status) 
and systolic blood pressure (BP). It employs machine learning techniques to build predictive models and determine the 
most accurate approach for predicting systolic BP.

---

Key Components:

1. Polynomial Regression:
   - Goal: Identify the optimal polynomial degree for predicting systolic BP based on serum cholesterol levels.
   - Implementation:
     - Used PolynomialFeatures from sklearn to generate higher-degree features.
     - Conducted 10-fold cross-validation to compute RMSE for polynomial degrees (1 to 14).
     - Visualised RMSE trends across degrees to select the best-performing model.
   - Outcome:
     - Determined the optimal polynomial degree based on the lowest RMSE.
     - Trained a polynomial regression model and extracted feature coefficients.

2. Multiple Linear Regression:
   - Goal: Predict systolic BP using all available patient features.
   - Implementation:
     - Features used: AGE, EDUCATION LEVEL, SMOKING STATUS, EXERCISE, WEIGHT, SERUM CHOLESTEROL, IQ, and SODIUM.
     - Trained a multiple linear regression model and evaluated it using 10-fold cross-validation.
     - Calculated the mean RMSE and analysed feature-specific coefficients.
   - Outcome:
     - Highlighted the contribution of individual features to systolic BP prediction.

3. Ridge Regression:
   - Goal: Apply regularisation to reduce overfitting and improve model robustness.
   - Implementation:
     - Used Ridge Regression with a regularisation parameter (alpha = 0.1).
     - Evaluated the model with 10-fold cross-validation and calculated mean RMSE.
     - Analysed coefficients after applying regularisation.
   - Outcome:
     - Demonstrated reduced overfitting and improved generalisation compared to standard linear regression.

---

Insights and Visualisations:
- RMSE trends across polynomial degrees were visualised, showing the relationship between degree and performance.
- Analysis revealed that serum cholesterol is a significant predictor of systolic BP.
- Ridge Regression enhanced model robustness by penalising large coefficients.

---

Significance:
This project bridges medical data and machine learning by:
- Demonstrating the importance of model selection (e.g., polynomial vs. multiple linear regression).
- Highlighting the value of regularisation (Ridge Regression) for robust predictions.
- Enhancing understanding of factors affecting blood pressure, with potential applications in healthcare 
  for personalised treatment recommendations.

---

Skills Highlighted:
- Tools/Technologies: Python, Pandas, NumPy, Scikit-learn, Matplotlib.
- Machine Learning Concepts: Cross-validation, Polynomial Regression, Ridge Regression, Model Evaluation Metrics.
- Visualisation and Analysis: RMSE plots and feature coefficient analysis for interpretability.
