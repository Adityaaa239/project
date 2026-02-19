# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2: Create dataset (Years of Experience vs Salary)
X = np.array([[1], [2], [3], [4], [5]])   # Input (2D array)
y = np.array([20000, 40000, 60000, 80000, 100000])  # Output

# Step 3: Create model
model = LinearRegression()

# Step 4: Train model
model.fit(X, y)

# Step 5: Predict salary for 6 years experience
prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])
