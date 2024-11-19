import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000
n_variables = 10  # Total variables including the target

# Generate independent variables (9 variables)
X = np.random.randn(n_samples, n_variables - 1)

# Define indices of the 3 variables that affect the target
affected_indices = [1, 4, 7]  # You can choose any 3 indices from 0 to 8

# Coefficients for the linear relationship
coefficients = np.array([2.5, -1.5, 3.0])

# Generate some noise
noise = np.random.normal(0, 0.5, n_samples)

# Compute the target variable
target = np.dot(X[:, affected_indices], coefficients) + noise

# Combine independent variables and target into a DataFrame
column_names = [f'Var{i+1}' for i in range(n_variables - 1)]  # Var1 to Var9
data = pd.DataFrame(X, columns=column_names)
data['Target'] = target

# Display the first few rows
print(data.head())
