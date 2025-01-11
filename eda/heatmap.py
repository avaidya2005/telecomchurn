import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into a pandas DataFrame
data = pd.read_csv('./datafile/orig/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data Preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce')
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# Drop rows with NaN values
data.dropna(inplace=True)

# Visualization 1: Distribution of Churn based on Contract Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=data)
plt.title('Distribution of Churn based on Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()

# Visualization 2: Correlation Heatmap
columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']
corr_matrix = data[columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Visualization 3: Monthly Charges Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['MonthlyCharges'], kde=True, bins=30)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()
