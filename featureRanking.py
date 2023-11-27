import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the heart data into a pandas DataFrame
heart_data = pd.read_csv('data_heart.csv')

# Separate the features (X) and the target variable (y)
X = heart_data.drop('DEATH_EVENT', axis=1)
X = X.drop('time', axis=1)
y = heart_data['DEATH_EVENT']

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the classifier to the data
rf.fit(X, y)

# Get the feature importances
importances = rf.feature_importances_

# Create a DataFrame with feature names and importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the features by importance in descending order
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Print the ranked features
print(feature_importances)