# Importing necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('OnlineArticlesPopularity.csv')



# Preprocessing: Fill missing values
# Assuming ' n_tokens_content' might have missing values and replacing 'Unknown' with NaN
data[' n_tokens_content'] = data[' n_tokens_content'].replace('Unknown', np.nan)
data[' n_tokens_content'] = data[' n_tokens_content'].fillna(data[' n_tokens_content'].mode()[0])

# Encoding categorical variables
# Assuming 'weekday' is a categorical variable that needs encoding
data = pd.get_dummies(data, columns=['weekday', 'channel type', 'isWeekEnd'])

# Drop unnecessary columns
# Assuming 'url' and ' timedelta' are unnecessary columns
data = data.drop(['url', ' timedelta'], axis=1)

# Remove non-numeric columns such as 'title'
data = data.select_dtypes(include=[np.number])

# Split the data into features and target
X = data.drop(' shares', axis=1)
y = data[' shares']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=r_regression, k=10)
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
# Get the selected feature names
selected_feature_names = X.columns[selector.get_support(indices=True)]

# Print the selected feature names
print("\nSelected feature names:", selected_feature_names)


# Get the feature names and their importances
feature_names = X.columns
feature_importances = selector.scores_

# Sort the features by their importance
sorted_indices = feature_importances.argsort()[::-1]
feature_names = feature_names[sorted_indices]
feature_importances = feature_importances[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), feature_importances, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()
# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train_selected, y_train)

# Random Forest Regressor Model
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train_selected, y_train)

# Save the models and preprocessing objects
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(lin_reg, file)
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_reg, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open('selector.pkl', 'wb') as file:
    pickle.dump(selector, file)

# Load the models and preprocessing objects
with open('linear_regression_model.pkl', 'rb') as file:
    lin_reg_loaded = pickle.load(file)
with open('random_forest_model.pkl', 'rb') as file:
    rf_reg_loaded = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
with open('selector.pkl', 'rb') as file:
    selector_loaded = pickle.load(file)

# Transform the test set
X_test_scaled = scaler_loaded.transform(X_test)
X_test_selected = selector_loaded.transform(X_test_scaled)

# Make predictions using the loaded models
y_pred_lin_reg_test = lin_reg_loaded.predict(X_test_selected)
y_pred_rf_reg_test = rf_reg_loaded.predict(X_test_selected)

# Calculate and print the Mean Squared Error for comparison
print('Linear Regression Test MSE:', mean_squared_error(y_test, y_pred_lin_reg_test))
print('Random Forest Regressor Test MSE:', mean_squared_error(y_test, y_pred_rf_reg_test))

# Calculate and print the R-squared value for comparison
print('Linear Regression Test R-squared:', r2_score(y_test, y_pred_lin_reg_test))
print('Random Forest Regressor Test R-squared:', r2_score(y_test, y_pred_rf_reg_test))
plt.scatter(y_test, y_pred_lin_reg_test)
plt.xlabel('Actual Values (Y_test)')
plt.ylabel('Predicted Values (Y_pred)')
plt.title('Random Forest Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.title('Linear Regression Model')
plt.show()
plt.scatter(y_test, y_pred_rf_reg_test)
plt.xlabel('Actual Values (Y_test)')
plt.ylabel('Predicted Values (Y_pred)')
plt.title('Random Forest Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.title('Random Forest Regresstion Model')
plt.show()