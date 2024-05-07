import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import tkinter as tk

# Load data
label_file = pd.read_csv('labels.csv')
data_file = pd.read_csv('data.csv')

# Merge label and data on sample column
merged_df = pd.merge(label_file, data_file, left_on='Sample', right_on='Unnamed: 0').drop(columns='Unnamed: 0')

# Extract features and target
X = merged_df.drop(['Sample', 'disease_type'], axis=1)
y = merged_df['disease_type']

X_normalized = X.div(X.sum(axis=1), axis=0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.1822, random_state=14)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate Random Forest classifier
rf_accuracy = rf_classifier.score(X_test, y_test)
rf_conf_matrix = confusion_matrix(y_test, rf_classifier.predict(X_test))

rf_sensitivity = rf_conf_matrix[0, 0] / (rf_conf_matrix[0, 0] + rf_conf_matrix[0, 1])
rf_specificity = rf_conf_matrix[1, 1] / (rf_conf_matrix[1, 0] + rf_conf_matrix[1, 1])

# Create tkinter window
root = tk.Tk()
root.geometry('400x400')
root.title("Introduction to Machine Learning Project")
root.configure(background="#e92929")

# Display results in tkinter window
results_label = tk.Label(root,text ="Results",font ="Verdana 16 bold")
results_label.pack()

accuracy_label = tk.Label(root, text=f"Accuracy: {rf_accuracy}")
accuracy_label.pack()

sensitivity_label = tk.Label(root, text=f"Sensitivity: {rf_sensitivity}")
sensitivity_label.pack()

specificity_label = tk.Label(root, text=f"Specificity: {rf_specificity}")
specificity_label.pack()

# Run tkinter event loop
root.mainloop()
