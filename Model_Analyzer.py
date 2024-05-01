from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pandas as pd
import numpy as np
from DataProcessor import item_data  # imports the processed data

# Loading and preprocess the dataset
X = item_data[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']]
y_cost = item_data['Cost']
y_efficiency = item_data['GoldEfficiency']

# Spliting the dataset into training and testing sets
X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
X_train, X_test, y_efficiency_train, y_efficiency_test = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)

#Standardize and Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters grid for tuning
param_grid = {
    'n_estimators': [25, 50, 75,100],
    'max_depth': [None,5, 10, 15,20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':['sqrt','log2',1,3,5,7,11],
    'max_leaf_nodes':[None,2,4,8]
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
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    accuracy = 100 * (1-mape) 
    print('Mean Absolute Error (MAE):', mse)
    print('Mean Squared Error (MSE):', mae)
    print('Root Mean Squared Error (RMSE):',rmse)
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(accuracy, 2))
        
    return best_model, mse,rmse,mae,mape

# Training and evaluating the model for cost prediction
print("Cost Prediction:")
best_model_cost, mse_cost,rmse_cost,mae_cost,mape_cost = train_and_evaluate(X_train, X_test, y_cost_train, y_cost_test)
print("Using Model: ",best_model_cost)

# Training and evaluating the model for efficiency prediction
best_model_efficiency, mse_efficiency,rmse_efficiency,mae_efficiency,mape_efficiency = train_and_evaluate(X_train, X_test, y_efficiency_train, y_efficiency_test)
print("\nEfficiency Prediction:")
print("Using Model: ",best_model_efficiency)
