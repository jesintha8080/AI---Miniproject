# AI---Miniproject
### BIOMARKERS FOR PREDICTING SPORTS PERFORMANCE
## Date - 23-10-2024
## Register Number : 212222040072
## Aim
To build and evaluate a machine Learning model for Predicting a target outcome based on given input features
## Algorithm
Step 1: Import necessary libraries

Step 2: Load the dataset and clean and normalize it for analysis

Step 3: Choose an appropriate Machine Learning model

Step 4: Train the selected model using the training dataset

Step 5: Evaluate the model performance using metrics such as accuracy,precision,recall.

Step 6: Interpret the model's prediction and visualizethe outcomes with relevant Graphs

## program

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('biomarkers_performance_data.csv')

print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Information about dataset columns and data types
print(data.describe())  # Summary statistics

data = data.dropna()  # Drop rows with missing values (or use data.fillna() to fill missing values)


X = data.drop(columns=['performance_metric'])  # All biomarkers as features
y = data['performance_metric']  # The target performance metric



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2 Score): {r2}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs Predicted Performance')
plt.show()

## Output
![image](https://github.com/user-attachments/assets/8b464ad9-41e5-41c0-a573-e120d7c1f7dc)
![image](https://github.com/user-attachments/assets/0047e782-2e3f-4843-a621-6e8d7f959fba)
![image](https://github.com/user-attachments/assets/f14e1a7e-aff4-4af2-87a1-9421aab5520e)

## Result
This is a final statement and output for Biomarkers for pedicting a sports performance.
