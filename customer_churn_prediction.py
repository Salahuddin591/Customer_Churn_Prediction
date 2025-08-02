# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load and explore the dataset ---
# NOTE: Make sure the 'WA_Fn-UseC_-Telco-Customer-Churn.csv' file is in the same directory.
print("--- Loading and exploring the dataset ---")
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Please download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle and place it in the same directory.")
    # Exiting the script if the file is not found
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get a summary of the data
print("\nDataset information:")
print(df.info())

# Drop 'customerID' as it's an identifier and not useful for prediction
df = df.drop('customerID', axis=1)

# --- 2. Data Cleaning and Preprocessing ---
print("\n--- Cleaning and preprocessing data ---")

# The 'TotalCharges' column is of type object, but contains numbers. It also has empty strings.
# We need to handle these empty strings and convert the column to numeric.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values (which are now NaNs from the conversion)
print("\nMissing values before handling:")
print(df.isnull().sum())

# We can fill the few missing 'TotalCharges' values with the mean, or drop the rows.
# Since there are only a few, dropping them is a simple and effective approach.
df.dropna(inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Use LabelEncoder to convert binary categorical columns (Yes/No, Male/Female) to 0 and 1.
# This makes them suitable for a machine learning model.
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' or x == 'Male' else 0)

# Convert other categorical features using one-hot encoding
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale numerical features to a standard range
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Performing Exploratory Data Analysis (EDA) ---")

# Plot churn rate by contract type
plt.figure(figsize=(8, 6))
sns.barplot(x=df['Contract_One year'], y=df['Churn'], hue=df['Contract_Two year'])
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type (0=Month-to-month, 1=One Year, 2=Two Year)')
plt.ylabel('Churn Rate')
plt.show()

# Plot distribution of MonthlyCharges for churned vs. non-churned customers
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, bins=30)
plt.title('MonthlyCharges Distribution by Churn Status')
plt.show()

# --- 4. Model Building and Training ---
print("\n--- Building and training the model ---")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# --- 5. Model Evaluation ---
print("\n--- Evaluating the model ---")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- 6. Feature Importance (Bonus) ---
print("\n--- Analyzing feature importance ---")

# Get feature importances from the trained model
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10])
plt.title('Top 10 Most Important Features for Churn Prediction')
plt.show()