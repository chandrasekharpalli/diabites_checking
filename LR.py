import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
diabetes = load_diabetes()
dataset = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
dataset['target'] = diabetes.target

# Features and target
X = dataset.iloc[:, :-1]
y = dataset['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# Scaling
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Train Linear Regression
Lreg = LinearRegression()
Lreg.fit(X_train, y_train)

# Predictions
y_pred = Lreg.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R² Score:", r2)

# Save scaler & model
pickle.dump(scalar, open('scaling.pkl', 'wb'))
pickle.dump(Lreg, open('regmodel.pkl', 'wb'))

# Save dataset to JSON
data_json = dataset.to_dict(orient="records")
with open("dataset.json", "w") as f:
    json.dump(data_json, f, indent=4)

# Load model and predict for first row
pickel_model = pickle.load(open('regmodel.pkl', 'rb'))
scaled_input = scalar.transform([dataset.iloc[0, :-1].values])
prediction = pickel_model.predict(scaled_input)
print("Prediction for first row:", prediction)

# Optional: Visualization
plt.scatter(y_pred, y_test)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual")
plt.show()

sns.displot(residuals, kind="kde")
plt.title("Residuals Distribution")
plt.show()