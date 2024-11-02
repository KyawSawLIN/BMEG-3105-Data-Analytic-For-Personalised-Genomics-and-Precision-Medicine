from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import time


df = pd.read_csv('data_heart.csv')
print(df.columns)
# Extract the features and target variable
X = df.drop('DEATH_EVENT', axis=1)  # Features (independent variables)
y = df['DEATH_EVENT']  # Target variable (dependent variable)


#################################
# Logistic Regression Classifier one time
#################################
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)



#################################
# Logistic Regression Classifier with cross validation
#################################

start_time = time.time()
cv = KFold(n_splits=50, shuffle=True, random_state=42)
# Compute cross-validated accuracy scores
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
f1 = cross_val_score(model, X, y, scoring='f1', cv=cv)
precision = cross_val_score(model, X, y, scoring='precision', cv=cv)
recall = cross_val_score(model, X, y, scoring='recall', cv=cv)
end_time = time.time()

print("Time taken:", end_time - start_time)
print("Mean Accuracy:", accuracy.mean())
print("Mean F1 score:", f1.mean())
print("Mean Precision score:", precision.mean())
print("Mean Recall score:", recall.mean())
  

