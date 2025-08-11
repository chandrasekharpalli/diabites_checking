# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
dataset = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
dataset['target'] = diabetes.target  # add target column

# Features (X) and Target (y)
X = dataset.iloc[:, :-1]
y = dataset['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression model
Lreg = LinearRegression()
Lreg.fit(X_train, y_train)

# Coefficients and Parameters
print("Coefficients:", Lreg.coef_)
print("Parameters:", Lreg.get_params())

# Predictions
y_pred = Lreg.predict(X_test)

# Scatter plot of predicted vs actual
plt.scatter(y_pred, y_test, alpha=0.6)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual")
plt.show()
# Residuals
residuals = y_test - y_pred
sns.displot(residuals, kind="kde")
plt.title("Residual Distribution")
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", score)