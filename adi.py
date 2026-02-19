# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression

# Step 2: Create dataset
# [Glucose level, BMI]
X = np.array([
    [85, 22],
    [90, 25],
    [150, 35],
    [130, 30],
    [70, 20],
    [160, 40]
])

# Output (0 = No diabetes, 1 = Diabetes)
y = np.array([0, 0, 1, 1, 0, 1])

# Step 3: Create model
model = LogisticRegression()

# Step 4: Train model
model.fit(X, y)

# Step 5: Predict for new person
# Example: Glucose=140, BMI=33
prediction = model.predict([[140, 33]])

print("Diabetes Prediction (0=No, 1=Yes):", prediction[0])
