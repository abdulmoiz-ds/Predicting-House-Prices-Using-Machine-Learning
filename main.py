# Section 1: Importing Libraries and Loading Data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
housing_data = pd.read_csv("housing_data.csv")

# Section 2: Data Cleaning and Preprocessing
# Drop any rows with missing data
housing_data.dropna(inplace=True)

# Split the data into features and target variable
X = housing_data.drop("price", axis=1)
y = housing_data["price"]

# Section 3: Feature Engineering
# Create dummy variables for categorical features
X = pd.get_dummies(X)

# Section 4: Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 5: Training the Model
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Section 6: Evaluating the Model
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the performance of the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Mean squared error:", mse)
print("R-squared:", r2)

# Section 7: Making Predictions on New Data
# Create a new observation for prediction
new_observation = [[2300, 3, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Make a prediction on the new observation
new_prediction = model.predict(new_observation)

# Print the predicted price
print("Predicted price:", new_prediction)

# Section 8: Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}

# Create a Random Forest Regression model
model = RandomForestRegressor(random_state=42)

# Create a grid search object
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)

# Perform the grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Section 9: Model Interpretation
# Calculate feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()
