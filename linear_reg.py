import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (ensure the correct path is provided)
data = pd.read_csv('modified_reorganized_dataset.csv')  # Update the path if necessary

# Display the first few rows of the dataset
print("Dataset preview:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check value counts for 'Glucose Category'
print("\nValue counts for 'Glucose Category':")
print(data['Glucose Category'].value_counts())

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Age Group', 'Glucose Category']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("\nEncoded Data:")
print(data.head())

# Separate features (X) and target variable (y)
X = data.drop('Glucose Category', axis=1)
y = data['Glucose Category']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled features prepared.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("\nLinear Regression model training complete.")

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Generate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Print training and testing accuracy
train_score = lr_model.score(X_train, y_train)
test_score = lr_model.score(X_test, y_test)
print(f"\nTrain R-squared: {train_score:.2f}")
print(f"Test R-squared: {test_score:.2f}")

# Save the model, scaler, and label encoders to .pkl files
joblib.dump(lr_model, 'glucose_lr_model.pkl')
joblib.dump(scaler, 'glucose_scaler.pkl')
joblib.dump(label_encoders, 'glucose_label_encoders.pkl')

print("\nModel, scaler, and label encoders saved successfully!")
