from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from DataProcessor import dataitem  # imports the processed data

# Loading and preprocess the dataset
X = dataitem[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']]
y_cost = dataitem['Cost']
y_efficiency = dataitem['GoldEfficiency']

# Spliting the dataset into training and testing sets
X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
X_train, X_test, y_efficiency_train, y_efficiency_test = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define a function for model training and evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    
    # Print evaluation metrics
    print("Mean Squared Error:", mse)
    
    return best_model, mse

# Train and evaluate the model for cost prediction
print("Cost Prediction:")
best_model_cost, mse_cost = train_and_evaluate(X_train_scaled, X_test_scaled, y_cost_train, y_cost_test)

# Train and evaluate the model for efficiency prediction
print("\nEfficiency Prediction:")
best_model_efficiency, mse_efficiency = train_and_evaluate(X_train_scaled, X_test_scaled, y_efficiency_train, y_efficiency_test)

# Now you have the best models and their respective MSEs
