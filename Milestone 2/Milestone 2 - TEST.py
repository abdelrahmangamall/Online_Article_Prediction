import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load new test data
new_test_data = pd.read_csv('OnlineArticlesPopularity_classification_test.csv')
# Load the filse

# Correct column names and remove any extra spaces
new_test_data.columns = new_test_data.columns.str.strip()
le = LabelEncoder()
# Handling missing values
new_test_data.replace('Unknown', np.nan, inplace=True)

# Fill missing values for numerical columns with the mean
numerical_cols = ['n_tokens_content', 'n_unique_tokens', 'average_token_length']
for col in numerical_cols:
    new_test_data[col] = new_test_data[col].fillna(new_test_data[col].mean())

# Fill missing values for categorical columns with the mode
categorical_cols = ['weekday', 'channel type', 'isWeekEnd']
for col in categorical_cols:
    new_test_data[col] = new_test_data[col].fillna(new_test_data[col].mode()[0])
new_test_data['Article Popularity'] = le.fit_transform(new_test_data['Article Popularity'])
new_test_data['weekday'] = le.fit_transform(new_test_data['weekday'])
new_test_data['channel type'] = le.fit_transform(new_test_data['channel type'])
new_test_data['isWeekEnd'] = le.fit_transform(new_test_data['isWeekEnd'])
# Define features (x) and target (y)
y_test = new_test_data['Article Popularity']
new_test_data = new_test_data.drop(['Article Popularity', 'url', 'title'], axis=1)
def load_model(model_name):
    with open(f'{model_name}_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)
model1 = load_model('Random Forest')
model2 = load_model('AdaBoost')
model3 = load_model('SVM')


# Transform the new test data
new_test_data_scaled = scaler.transform(new_test_data)

# Make predictions using the loaded models
y_pred1 = model1.predict(new_test_data_scaled)
y_pred2 = model2.predict(new_test_data_scaled)
y_pred3 = model3.predict(new_test_data_scaled)

accuracy1 = accuracy_score(y_test, y_pred1)
print("Accuracy of Random Forest: ",accuracy1)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy of AdaBoost: ",accuracy2)
accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy of SVM: ",accuracy3)