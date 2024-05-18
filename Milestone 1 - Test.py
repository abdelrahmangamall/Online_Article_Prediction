# Importing necessary libraries
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# Load new test data
new_test_data = pd.read_csv('OnlineArticlesPopularity_test.csv')


y_test = new_test_data[' shares']
new_test_data = pd.get_dummies(new_test_data, columns=['weekday', 'channel type', 'isWeekEnd'])
new_test_data = new_test_data.drop(' shares', axis=1)
new_test_data = new_test_data.drop(['url', ' timedelta'], axis=1)
new_test_data = new_test_data.select_dtypes(include=[np.number])

# Load the models and preprocessing objects
with open('linear_regression_model.pkl', 'rb') as file:
    lin_reg_loaded = pickle.load(file)
with open('random_forest_model.pkl', 'rb') as file:
    rf_reg_loaded = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
with open('selector.pkl', 'rb') as file:
    selector_loaded = pickle.load(file)

# Transform the new test data
new_test_data_scaled = scaler_loaded.transform(new_test_data)
new_test_data_selected = selector_loaded.transform(new_test_data_scaled)

# Make predictions using the loaded models
y_pred_lin_reg_new_test = lin_reg_loaded.predict(new_test_data_selected)
y_pred_rf_reg_new_test = rf_reg_loaded.predict(new_test_data_selected)

# Output the predictions
# Calculate and print the Mean Squared Error for comparison
print('Linear Regression Test MSE:', mean_squared_error(y_test, y_pred_lin_reg_new_test))
print('Random Forest Regressor Test MSE:', mean_squared_error(y_test, y_pred_rf_reg_new_test))

# Calculate and print the R-squared value for comparison
print('Linear Regression Test R-squared:', r2_score(y_test, y_pred_lin_reg_new_test))
print('Random Forest Regressor Test R-squared:', r2_score(y_test, y_pred_rf_reg_new_test))