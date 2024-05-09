from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.impute import SimpleImputer
import numpy as np
from DataProcessor import item_data 
from time import time
import matplotlib.pyplot as plt


# Loading and preprocess the dataset
X = item_data[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']]
y_cost = item_data['Cost']
y_efficiency = item_data['GoldEfficiency']

# Splitting the dataset into training and testing sets
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X, y_cost, test_size=0.2, random_state=42)
X_train_efficiency, X_test_efficiency, y_train_efficiency, y_test_efficiency = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)

# Standardize and scale the features
scaler = StandardScaler()
X_train_cost_scaled = scaler.fit_transform(X_train_cost)
X_test_cost_scaled = scaler.transform(X_test_cost)
X_train_efficiency_scaled = scaler.fit_transform(X_train_efficiency)
X_test_efficiency_scaled = scaler.transform(X_test_efficiency)

# Filling missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train_cost_imputed = imputer.fit_transform(X_train_cost_scaled)
X_test_cost_imputed = imputer.transform(X_test_cost_scaled)
X_train_efficiency_imputed = imputer.fit_transform(X_train_efficiency_scaled)
X_test_efficiency_imputed = imputer.transform(X_test_efficiency_scaled)

# Define hyperparameters grid for tuning
param_grid = {
    'n_estimators': [25, 50, 75,100],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
} 

# Define a function for model training and evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    accuracy = 100 * (1 - mape) 
    
    # Print evaluation results
    print(f"{model_name} Model Evaluation:")
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(accuracy, 2))
    print('Best Parameters:', grid_search.best_params_)
        
    return best_model, mse, rmse, mae, mape

start_time = time()
# Training and evaluating the model for cost prediction
best_model_cost, mse_cost, rmse_cost, mae_cost, mape_cost = train_and_evaluate(X_train_cost_imputed, X_test_cost_imputed, y_train_cost, y_test_cost, "Cost")
end_time = time()
print("\nUsing Model:", best_model_cost)
print(f"It took {end_time-start_time} seconds to find the Model")

start_time = time()
# Training and evaluating the model for efficiency prediction
best_model_efficiency, mse_efficiency, rmse_efficiency, mae_efficiency, mape_efficiency = train_and_evaluate(X_train_efficiency_imputed, X_test_efficiency_imputed, y_train_efficiency, y_test_efficiency, "Efficiency")
end_time = time()
print("\nUsing Model:", best_model_efficiency)
print(f"It took {end_time-start_time} seconds to find the Model")

cost_y_pred = best_model_cost.predict(X_test_cost_imputed)
efficiency_y_pred = best_model_efficiency.predict(X_test_efficiency_imputed)

plt.figure(figsize=(10, 8))

# Plot predicted vs. actual values for cost prediction

plt.subplot(1, 2, 1)
plt.scatter(y_test_cost, cost_y_pred, color='blue')
plt.plot([min(y_test_cost), max(y_test_cost)], [min(y_test_cost), max(y_test_cost)], '--', color='green')
plt.title("Predicted vs Actual (Cost)")
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_efficiency, efficiency_y_pred, color='red')
plt.plot([min(y_test_efficiency), max(y_test_efficiency)], [min(y_test_efficiency), max(y_test_efficiency)], '--', color='green')
plt.title("Predicted vs Actual (Efficiency)")
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.grid(True)

plt.tight_layout()
plt.show()
