import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the data
data = pd.read_csv('datafile/orig/telcochurn.csv')

# Handle missing values
data = data.fillna('')

# Convert categorical variables to numeric
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
data['tenure'] = data['tenure'].astype(int)
data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)
data['TotalCharges'] = data['TotalCharges'].replace(' ', 0).astype(float)

# Exclude 'CustomerID' before applying get_dummies
customer_ids = data['CustomerID']
data = data.drop(columns=['CustomerID'])

# Encode categorical features
data = pd.get_dummies(data)

# Add 'CustomerID' back to the DataFrame
data['CustomerID'] = customer_ids

# Ensure 'CustomerID' and 'Churn' columns are dropped from features
features = data.drop(columns=['CustomerID', 'Churn_Yes', 'Churn_No'])
target = data['Churn_Yes']

# Print column names to verify

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# Get feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': np.abs(model.coef_[0])
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
