import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Load your data into a pandas DataFrame
data = pd.read_csv('./datafile/orig/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data Preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce')
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
data['Partner'] = data['Partner'].map({'No': 0, 'Yes': 1})
data['Dependents'] = data['Dependents'].map({'No': 0, 'Yes': 1})
data['PhoneService'] = data['PhoneService'].map({'No': 0, 'Yes': 1})
data['MultipleLines'] = data['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 0})
data['InternetService'] = data['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
data['OnlineSecurity'] = data['OnlineSecurity'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['OnlineBackup'] = data['OnlineBackup'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['DeviceProtection'] = data['DeviceProtection'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['TechSupport'] = data['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['StreamingTV'] = data['StreamingTV'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['StreamingMovies'] = data['StreamingMovies'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'No': 0, 'Yes': 1})
data['PaymentMethod'] = data['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})

# Drop rows with NaN values
data.dropna(inplace=True)

# Select numeric columns for correlation matrix
numeric_columns = [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
    'TotalCharges', 'Churn'
]

# Calculate correlation matrix
corr_matrix = data[numeric_columns].corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Feature Importance using Random Forest
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.2f}')

feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

print('Feature Importances:')
print(feature_importances)

# Predictive Modeling using Logistic Regression
lr_model = LogisticRegression(max_iter=10000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))
