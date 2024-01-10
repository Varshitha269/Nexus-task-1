# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ask user for the CSV file input
file_path = input("Enter the path to the CSV file: ")

# Load the dataset
try:
    iris_df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Display the first few rows of the dataset
print(iris_df.head())

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Perform basic EDA
sns.pairplot(iris_df, hue="Species")
plt.show()

# Split the dataset into features (X) and target variable (y)
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
