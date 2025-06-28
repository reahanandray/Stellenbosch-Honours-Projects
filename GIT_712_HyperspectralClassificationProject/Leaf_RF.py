import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import matthews_corrcoef

# Define the path on local machine to the rawASD data
file_path = '/Users/reah/Downloads/D3_Resources/Data/Spectral Samples/ASD_raw_Dec2023_LA.csv'

#  Load the rawASD dataset
rawASD = pd.read_csv(file_path)

print("Leaf Dataset using Random Forest:")
print(".")

print("Column headers from list:", list(rawASD))
for x in rawASD.index:
    if 350 <= rawASD.loc[x, "Unnamed: 0"] <= 399 \
            or 1350 <= rawASD.loc[x, "Unnamed: 0"] <= 1425 \
            or 1825 <= rawASD.loc[x, "Unnamed: 0"] <= 1925 \
            or 2451 <= rawASD.loc[x, "Unnamed: 0"] <= 2500:
        rawASD.drop(x, inplace=True)

# Check the first few rows of the dataset to ensure it loaded correctly
print(rawASD.head())
print("Dataset dimensions: ", rawASD.shape)

# 1. Plot All Spectra with Atmospheric Windows

# Plot all spectra (excluding first column which contains wavelength)
plt.figure(figsize=(10, 6))
for col in rawASD.columns[1:]:
    plt.plot(rawASD.iloc[:, 0], rawASD[col], linewidth=1)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Sample Reflectance')
plt.title('Spectral Plot')
plt.show()

# 2. Clean and Reformat Data for Analysis

# Convert the rawASD DataFrame (transposed)
input_data = rawASD.T  # Transpose the DataFrame
input_data.columns = rawASD.iloc[:, 0]  # Set column names to first row
input_data.columns = ['wb_' + str(col) if isinstance(col, (int, float)) else 'wb_' + col for col in input_data.columns]

# Remove the first row (header) which is now the column names
input_data = input_data.drop(index=input_data.index[0])

# Check variable types
print(input_data.dtypes)

# Create 'Cultivar' class labels based on row names (index)
input_data['Cultivar'] = input_data.index

# Replace cultivar labels as in R's sub function
input_data['Cultivar'] = input_data['Cultivar'].replace({
    r'Sul\d{1,}': 'Sultana',
    r'Sug\d{1,5}': 'Sugra-39',
    r'Cur\d{1,5}': 'Currents'
}, regex=True)

# Ensure Cultivar is a categorical (factor-like) column
input_data['Cultivar'] = input_data['Cultivar'].astype('category')

# Check if 'Cultivar' is a factor (categorical column)
print(input_data['Cultivar'].dtype)

# Reset row names (index) to numeric
input_data.reset_index(drop=True, inplace=True)

# Summarize the number of samples by Cultivar
cultivar_summary = input_data['Cultivar'].value_counts()
print(cultivar_summary)

# ===============================================================================
#                 First iteration - Run Leaf data as it is with RF
# ===============================================================================

print("\n------ FIRST ITERATION -------")

# 1. SPLIT DATA INTO TRAIN AND TEST SET

# Define features (X) and target labels (y)
X = input_data.drop(columns=['Cultivar'])  # Features: All columns except 'Cultivar'
y = input_data['Cultivar']  # Target: Cultivar classification labels

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Print class mapping to label identify Cultivars
print("\nCultivar Class Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Split into train and test sets (70 - train, 30 - test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Print dataset shapes to verify
print("Total Training set size:", X_train.shape)
print("Total Testing set size:", X_test.shape)

# 2. TRAIN AND TEST ON TRAINING AND TEST DATA - RF Model

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# 3. PRODUCE ACCURACY MATRIX AND MCC FOR TRAIN AND TEST DATA

# Make predictions
y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

# Calculate accuracy and classification report train
train_accuracy = accuracy_score(y_train, y_train_pred)
train_classification_rep = classification_report(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_rep = classification_report(y_test, y_test_pred)

# Print the results
print(f"\nRandom Forest Training Set Accuracy: {train_accuracy:.2f}")
print("\nRandom Forest Training Classification Report:\n", train_classification_rep)

print(f"\nRandom Forest Test Set Accuracy: {test_accuracy:.2f}")
print("\nRandom Forest Test Classification Report:\n", test_classification_rep)

# Calculate MCC for train and test sets
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print MCC along with other metrics
print(f"Random Forest Training MCC: {mcc_train:.2f}")
print(f"Random Forest Test MCC: {mcc_test:.2f}")

# 4. CLASS IMBALANCE AT EVALUATION LEVEL - ROC AUC

# Test Set

# Use probabilities for ROC AUC
y_pred_prob = rf_classifier.predict_proba(X_test)

# Compute the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print('Random Forest ROC AUC Test Score :', roc_auc)

# ROC curve for Multi classes
colors = ['orange', 'red', 'green']
class_names = label_encoder.classes_
for i in range(len(class_names)):
    fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    plt.plot(fpr, tpr, linestyle='--', color=colors[i % len(colors)], label=f'{class_names[i]} vs Rest')

# ROC curve for tpr = fpr (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.title('Random Forest Grapevine Cultivar ROC Curve on Test Set')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 5. PRODUCE CONFUSION MATRIX FOR TRAIN AND TEST DATA

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred, labels=range(len(label_encoder.classes_)))  # Use integer labels
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=label_encoder.classes_)
disp_train.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Training Set)', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred, labels=range(len(label_encoder.classes_)))  # Use integer labels
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=label_encoder.classes_)
disp_test.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

# ===============================================================================
#         Second iteration - Dimensionality Reduction using Boruta Wrapper
# ===============================================================================

print("\n------ SECOND ITERATION --------")
print(".")

# 1. USE FEATURE SELECTION TO CREATE OPTIMAL SUBSET

# Initialize Boruta feature selection with RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)

# Perform feature selection on the training data
boruta_selector.fit(X_train.values, y_train)

# Get the selected features
selected_features = X_train.columns[boruta_selector.support_]
print("Selected Features:", list(selected_features))

# Calculate and print percentage of dimensionality reduction
original_feature_count = X_train.shape[1]
selected_feature_count = len(selected_features)
dimensionality_reduction = (1 - (selected_feature_count / original_feature_count)) * 100

print(f"Original Feature Count: {original_feature_count}")
print(f"Selected Feature Count: {selected_feature_count}")
print(f"Percentage of Dimensionality Reduction: {dimensionality_reduction:.2f}%")

# Transform training and testing datasets to keep only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 2. TRAIN AND TEST ON SUBSET TRAINING AND TEST DATA - RF Model

# Train Random Forest on the selected features
rf_classifier_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_selected.fit(X_train_selected, y_train)

# 3. PRODUCE ACCURACY MATRIX AND MCC FOR TRAIN AND TEST DATA

# Make predictions
y_train_pred_selected = rf_classifier_selected.predict(X_train_selected)
y_test_pred_selected = rf_classifier_selected.predict(X_test_selected)

# Calculate accuracy and classification report
train_accuracy_selected = accuracy_score(y_train, y_train_pred_selected)
test_accuracy_selected = accuracy_score(y_test, y_test_pred_selected)

train_classification_rep_selected = classification_report(y_train, y_train_pred_selected)
test_classification_rep_selected = classification_report(y_test, y_test_pred_selected)

# Print the results
print(f"\nRandom Forest with Boruta Feature Selection - Training Accuracy: {train_accuracy_selected:.2f}")
print("\nRandom Forest Train Classification Report (After Feature Selection):\n", train_classification_rep_selected)

print(f"\nRandom Forest with Boruta Feature Selection - Test Accuracy: {test_accuracy_selected:.2f}")
print("\nRandom Forest Test Classification Report (After Feature Selection):\n", test_classification_rep_selected)

# Calculate MCC for train and test sets
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print MCC along with other metrics
print(f"Random Forest Training MCC: {mcc_train:.2f}")
print(f"Random Forest Test MCC: {mcc_test:.2f}")

# 4. CLASS IMBALANCE AT EVALUATION LEVEL - ROC AUC

# Test Set

# Compute the ROC AUC score
y_pred_prob_selected = rf_classifier_selected.predict_proba(X_test_selected)
roc_auc_selected = roc_auc_score(y_test, y_pred_prob_selected, multi_class='ovr')
print('Random Forest ROC AUC Test Score (After Feature Selection):', roc_auc_selected)

# ROC curve for classes
colors = ['orange', 'red', 'green']
class_names = label_encoder.classes_
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_selected[:, i], pos_label=i)
    plt.plot(fpr, tpr, linestyle='--', color=colors[i % len(colors)], label=f'{class_names[i]} vs Rest')

plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.title('Random Forest ROC Curve on Test Set (After Feature Selection)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 5. PRODUCE CONFUSION MATRIX FOR TRAIN AND TEST DATA

# Confusion Matrix for Test Set
cm_test_selected = confusion_matrix(y_test, y_test_pred_selected, labels=range(len(label_encoder.classes_)))
disp_test_selected = ConfusionMatrixDisplay(confusion_matrix=cm_test_selected, display_labels=label_encoder.classes_)
disp_test_selected.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set) - After Feature Selection', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

# ===============================================================================
#          Third iteration - Account for Class Imbalance on a Data-Level
# ===============================================================================

print("\n------- THIRD ITERATION --------")

# 1. BALANCE THE SUBSET TRAIN DATA USING SMOTE

# Display class distribution before SMOTE
print("\nClass distribution before SMOTE:", Counter(y_train))

# Apply SMOTE to over-sample the minority class in the training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Display class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_train_resampled))

# 2. TRAIN AND TEST ON RESAMPLED SUBSET TRAINING AND TEST DATA - RF Model

# Train Random Forest on the resampled training data
rf_classifier_resampled = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_resampled.fit(X_train_resampled, y_train_resampled)

# 3. PRODUCE ACCURACY MATRIX AND MCC FOR TRAIN AND TEST DATA

# Make predictions
y_train_pred_resampled = rf_classifier_resampled.predict(X_train_resampled)
y_test_pred_resampled = rf_classifier_resampled.predict(X_test_selected)

# Calculate accuracy and classification report
train_accuracy_resampled = accuracy_score(y_train_resampled, y_train_pred_resampled)
test_accuracy_resampled = accuracy_score(y_test, y_test_pred_resampled)
test_classification_rep_resampled = classification_report(y_test, y_test_pred_resampled)
train_classification_rep_resampled = classification_report(y_train_resampled, y_train_pred_resampled)

# Print the results
print(f"\nRandom Forest Training Set Accuracy (After SMOTE): {train_accuracy_resampled:.2f}")
print("\nRandom Forest Training Classification Report (After SMOTE):\n", train_classification_rep_resampled)
print(f"\nRandom Forest Test Set Accuracy (After SMOTE): {test_accuracy_resampled:.2f}")
print("\nRandom Forest Test Classification Report (After SMOTE):\n", test_classification_rep_resampled)

# Calculate MCC for train and test sets
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print MCC along with other metrics
print(f"Random Forest Training MCC: {mcc_train:.2f}")
print(f"Random Forest Test MCC: {mcc_test:.2f}")

# 4. CLASS IMBALANCE AT EVALUATION LEVEL - ROC AUC

# Test Set

# Compute the ROC AUC score
y_pred_prob_resampled = rf_classifier_resampled.predict_proba(X_test_selected)
roc_auc_resampled = roc_auc_score(y_test, y_pred_prob_resampled, multi_class='ovr')
print('Random Forest ROC AUC Test Score (After SMOTE):', roc_auc_resampled)

# ROC curve for classes
colors = ['orange', 'red', 'green']
class_names = label_encoder.classes_

for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_resampled[:, i], pos_label=i)
    plt.plot(fpr, tpr, linestyle='--', color=colors[i % len(colors)], label=f'{class_names[i]} vs Rest')

plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.title('Random Forest ROC Curve on Test Set (After SMOTE)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 5. PRODUCE CONFUSION MATRIX FOR TRAIN AND TEST DATA

# Confusion Matrix for Test Set
cm_test_resampled = confusion_matrix(y_test, y_test_pred_resampled, labels=range(len(label_encoder.classes_)))
disp_test_resampled = ConfusionMatrixDisplay(confusion_matrix=cm_test_resampled, display_labels=label_encoder.classes_)
disp_test_resampled.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set) - After SMOTE', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

