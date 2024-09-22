# Diabetes, Obesity, and Inactivity: A Multi-Linear Regression Analysis

## Overview

This project investigates the relationships between diabetes, obesity, and inactivity across various U.S. counties in 2018. We use a multi-linear regression model to analyze the data and attempt to predict diabetes rates based on inactivity and obesity data.

## Author
- Suraj Basavaraj Rajalod - 02131154

## Project Objective

The key goals of this project include:
- Predicting diabetes rates based on inactivity data using a linear regression model.
- Investigating the correlation between diabetes, obesity, and inactivity data through statistical plots and multi-linear regression.
- Performing the Breusch-Pagan test to evaluate heteroscedasticity in the model's residuals.

## Data

The dataset includes:
- FIPS (unique identifier for U.S. counties)
- County name
- State name
- Year (2018)
- Percentages of people diagnosed with diabetes, obesity, and inactivity.

## Key Findings

- **Correlation Analysis:** Diabetic data is more closely correlated with inactivity than obesity. We used Pearson's correlation to confirm this.
- **Regression Models:** Both linear and multi-linear regression models were implemented. The linear model uses inactivity data to predict diabetes, while the multi-linear model uses both inactivity and obesity data.
- **Heteroscedasticity Test:** The Breusch-Pagan test indicates that heteroscedasticity is not significant (p-value > 0.05), suggesting the model is reliable.

## Methodology

1. **Data Loading:** The dataset was loaded into Python data frames for analysis.
2. **Exploratory Data Analysis (EDA):** We examined the distribution and correlation of the variables using pair plots, histograms, and heatmaps.
3. **Linear Regression:** Implemented using inactivity as the independent variable and diabetes as the dependent variable.
4. **Multi-Linear Regression:** Extended the model to include both inactivity and obesity as predictors of diabetes.
5. **Model Evaluation:** Evaluated the models using R-squared values, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
6. **Breusch-Pagan Test:** Conducted to check for heteroscedasticity in the model's residuals.

### Sample Code for Regression Models

- **Linear Regression:** Predicting diabetes using inactivity data.
  
```python
# Example: Linear Regression with Inactivity Data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split the data
X_train, X_test, y_train, y_test = train_test_split(inactivity_data, diabetic_data, test_size=0.25)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Example: Multi-Linear Regression with Inactivity and Obesity Data
X = merged_data[['%Inactivity', '%Obesity']]
y = merged_data['%Diabetic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the model
ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

# Test the model
predictions = ml_model.predict(X_test)
print(f"R-squared: {ml_model.score(X_test, y_test)}")
```
## Results

### Pair Plot
![Alt text](/heatgraph.jpg)

### Histogram
![Alt text](/Bargraph.jpg)

### Correlation Heatmap
![Alt text](/heatmap.jpg)

### Scatter Plot with Regression Line
![Alt text](/Linear.jpg)


## Requirements
- Python 3.x
- Jupyter Notebook
- Required Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Conclusion

The regression models provided useful insights into the relationship between diabetes, obesity, and inactivity. The findings indicate that inactivity is a stronger predictor of diabetes rates than obesity, and the multi-linear regression model further supports this correlation.
