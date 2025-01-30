import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train an SVM Classifier
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
print("\nSVM model training complete.")

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['Glucose Category'].classes_)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute True Positives, False Positives, True Negatives, False Negatives
true_positives = np.diag(cm)
false_positives = cm.sum(axis=0) - true_positives
false_negatives = cm.sum(axis=1) - true_positives
true_negatives = cm.sum() - (true_positives + false_positives + false_negatives)

# Display per-class metrics
for i, class_name in enumerate(label_encoders['Glucose Category'].classes_):
    print(f"\nClass: {class_name}")
    print(f"  True Positives: {true_positives[i]}")
    print(f"  False Positives: {false_positives[i]}")
    print(f"  False Negatives: {false_negatives[i]}")
    print(f"  True Negatives: {true_negatives[i]}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=label_encoders['Glucose Category'].classes_, 
            yticklabels=label_encoders['Glucose Category'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print training and testing accuracy
train_accuracy = svm_model.score(X_train, y_train)
test_accuracy = svm_model.score(X_test, y_test)
print(f"\nTrain accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Save the model, scaler, and label encoders to .pkl files
joblib.dump(svm_model, 'glucose_svm_model.pkl')
joblib.dump(scaler, 'glucose_scaler.pkl')
joblib.dump(label_encoders, 'glucose_label_encoders.pkl')

print("\nModel, scaler, and label encoders saved successfully!")
