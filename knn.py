from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# Define the range of K values to try
k_values = [3, 5, 7, 9, 11]

df = pd.read_csv('data_heart.csv')
# Extract the features and target variable
X = df.drop('DEATH_EVENT', axis=1).values  # Features (independent variables)
y = df['DEATH_EVENT'].values  # Target variable (dependent variable)

best_accuracy = 0
best_k = None

# Perform cross-validation for each K value
for k in k_values:
    # Create a KNN classifier with the current K value
    knn = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation using KFold
    cv = KFold(n_splits=50, shuffle=True, random_state=42)

    # Compute cross-validated accuracy scores
    scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv)

    # Calculate the average accuracy score
    average_accuracy = scores.mean()

    # Check if this K value gives a better accuracy
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_k = k

# Print the best K value
print(f"Best K: {best_k}")

# Create a KNN classifier with the best K value
knn = KNeighborsClassifier(n_neighbors=best_k)
cv = KFold(n_splits=50, shuffle=True, random_state=42)
accuracy = cross_val_score(knn, X, y, scoring='accuracy', cv=cv)
f1 = cross_val_score(knn, X, y, scoring='f1', cv=cv)
precision = cross_val_score(knn, X, y, scoring='precision', cv=cv)
recall = cross_val_score(knn, X, y, scoring='recall', cv=cv)

print("Mean Accuracy:", accuracy.mean())
print("Mean F1 score:", f1.mean())
print("Mean Precision score:", precision.mean())
print("Mean Recall score:", recall.mean())
  

