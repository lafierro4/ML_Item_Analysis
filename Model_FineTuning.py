from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from DataProcessor import item_data
import matplotlib.pyplot as plt

# Loading and preprocess the dataset
X = item_data[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']]
y_cost = item_data['Cost']
y_efficiency = item_data['GoldEfficiency']

# Spliting the dataset into training and testing sets
X_train_cost, X_test_cost, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
X_train_efficiency, X_test_efficiency, y_efficiency_train, y_efficiency_test = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)

# Standardize and scale the features
scaler = StandardScaler()
X_train_cost_scaled = scaler.fit_transform(X_train_cost)
X_test_cost_scaled = scaler.transform(X_test_cost)
X_train_efficiency_scaled = scaler.fit_transform(X_train_efficiency)
X_test_efficiency_scaled = scaler.transform(X_test_efficiency)


#Using the best models that were found and training them on the data
cost_model = GradientBoostingRegressor(learning_rate=0.2, max_depth=4, max_features='log2',
                          min_samples_leaf=2, min_samples_split=4,
                          n_estimators=75)
cost_model.fit(X_train_cost_scaled,y_cost_train)

efficiency_model =  GradientBoostingRegressor(learning_rate=0.2, max_depth=4, max_features='sqrt',
                          min_samples_split=3, n_estimators=75)
efficiency_model.fit(X_train_efficiency_scaled,y_efficiency_train)

#Making predictions on the models using testing data:
cost_y_pred = cost_model.predict(X_test_cost_scaled)
efficiency_y_pred = efficiency_model.predict(X_test_efficiency_scaled)


#Evaluating Models
cost_mse = metrics.mean_squared_error(y_cost_test, cost_y_pred)
efficiency_mse = metrics.mean_squared_error( y_efficiency_test, efficiency_y_pred) 

cost_rmse = np.sqrt(metrics.mean_squared_error(y_cost_test,cost_y_pred))
efficiency_rmse = np.sqrt(metrics.mean_squared_error( y_efficiency_test,efficiency_y_pred))

cost_mae = metrics.mean_absolute_error(y_cost_test, cost_y_pred)
efficiency_mae = metrics.mean_absolute_error( y_efficiency_test, efficiency_y_pred)

cost_mape = np.mean(np.abs((y_cost_test - cost_y_pred) / np.abs(y_cost_test)))
efficiency_mape = np.mean(np.abs(( y_efficiency_test - efficiency_y_pred) / np.abs( y_efficiency_test)))


# Print Model Performance Metrics
print("\nGold Cost Model:")
print('Mean Absolute Error (MAE):', cost_mse)
print('Mean Squared Error (MSE):', cost_mae)
print('Root Mean Squared Error (RMSE):',cost_rmse)
print('Mean Absolute Percentage Error (MAPE):', round(cost_mape * 100, 2))
print('Accuracy:', round(100*(1 - cost_mape), 2))

print("\nGold Efficency Model:")
print('Mean Absolute Error (MAE):', efficiency_mse)
print('Mean Squared Error (MSE):', efficiency_mae)
print('Root Mean Squared Error (RMSE):',efficiency_rmse)
print('Mean Absolute Percentage Error (MAPE):', round(efficiency_mape * 100, 2))
print('Accuracy:', round(100*(1 - efficiency_mape), 2))

# Predicted vs. Actual Values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_cost_test, cost_y_pred, color='blue')
plt.title("Predicted vs Actual (Cost)")
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")

plt.subplot(1, 2, 2)
plt.scatter(y_efficiency_test, efficiency_y_pred, color='red')
plt.title("Predicted vs Actual (Efficiency)")
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")

plt.tight_layout()
plt.show()




