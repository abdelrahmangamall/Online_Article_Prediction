import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("OnlineArticlesPopularity_Milestone2.csv")

# Correct column names and remove any extra spaces
data.columns = data.columns.str.strip()


# Handling missing values
data.replace('Unknown', np.nan, inplace=True)

# Fill missing values for numerical columns with the mean
numerical_cols = ['n_tokens_content', 'n_unique_tokens', 'average_token_length']
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())

# Fill missing values for categorical columns with the mode
categorical_cols = ['weekday', 'channel type', 'isWeekEnd']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encoding categorical data
le = LabelEncoder()
data['weekday'] = le.fit_transform(data['weekday'])
data['channel type'] = le.fit_transform(data['channel type'])
data['isWeekEnd'] = le.fit_transform(data['isWeekEnd'])

# Save the label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump({'weekday': le, 'channel type': le, 'isWeekEnd': le}, file)

# Encode the target variable
data['Article Popularity'] = le.fit_transform(data['Article Popularity'])

# Define features (x) and target (y)
x = data.drop(['Article Popularity', 'url', 'title'], axis=1)
y = data['Article Popularity']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, algorithm='SAMME'),
    'SVM': SVC(kernel='linear', C=1, gamma=0.1)
}

# Dictionary to hold training and testing times
times = {name: {'Training Time': 0, 'Testing Time': 0} for name in classifiers.keys()}

# Train classifiers, evaluate them, and save the models
for name, clf in classifiers.items():
    start_train_time = time.time()
    clf.fit(x_train, y_train)
    end_train_time = time.time()
    times[name]['Training Time'] = end_train_time - start_train_time

    start_test_time = time.time()
    y_pred = clf.predict(x_test)
    end_test_time = time.time()
    times[name]['Testing Time'] = end_test_time - start_test_time

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.3f}")
    print(f"Mean Square Error of {name}: {mse:.3f}")
    print(f"Training Time of {name}: {times[name]['Training Time']:.3f} seconds")
    print(f"Testing Time of {name}: {times[name]['Testing Time']:.3f} seconds")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    with open(f'{name}_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

# Plotting the results
metrics = ['Training Time', 'Testing Time', 'Accuracy']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    classifier_names = list(times.keys())
    metric_values = [times[classifier][metric] if metric != 'Accuracy' else accuracy_score(y_test, classifiers[classifier].predict(x_test)) for classifier in classifier_names]
    plt.bar(classifier_names, metric_values)
    plt.title(f'Comparison of {metric}')
    plt.ylabel(metric)
    plt.show()

# Load the model and the scaler
def load_model_and_scaler(model_name):
    with open(f'{model_name}_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Load the preprocessing label encoders
def load_label_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

# Example of loading a model and scaler
model, scaler = load_model_and_scaler('Random Forest')

# Example of loading label encoders
encoders = load_label_encoders()

# Preprocess and predict on new test data
def preprocess_and_predict(test_data, model, scaler, encoders):
    # Preprocess the test data in the same way as the training data
    test_data = test_data.copy()
    test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
    for col in categorical_cols:
        test_data[col] = encoders[col].transform(test_data[col])
    
    # Predict using the loaded model
    predictions = model.predict(test_data)
    return predictions

# Example usage with new test data
new_test_data = pd.read_csv("OnlineArticlesPopularity_Milestone2.csv")
predictions = preprocess_and_predict(new_test_data, model, scaler, encoders)