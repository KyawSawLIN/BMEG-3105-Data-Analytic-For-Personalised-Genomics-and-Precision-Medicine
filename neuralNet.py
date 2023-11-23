import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

# Assuming you have the heart failure dataset stored in a pandas DataFrame named 'df'

# Define a custom dataset class
class HeartFailureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

df = pd.read_csv('data_heart.csv')
# Extract the features and target variable
X = df.drop('DEATH_EVENT', axis=1).values  # Features (independent variables)
y = df['DEATH_EVENT'].values  # Target variable (dependent variable)

# Scale the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Create an instance of the neural network model
model = NeuralNetwork(input_size=X.shape[1])

# Define a scikit-learn compatible wrapper for the neural network model
class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = [0, 1]  # Dummy classes_ attribute for binary classification

    
    def fit(self, X, y):
        # Convert X and y to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Define a custom data loader for training
        dataset = HeartFailureDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train the model
        self.model.train()
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        # Convert X to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            predictions = (output >= 0.5).squeeze().numpy().astype(int)
        
        return predictions

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)

# Create an instance of the scikit-learn compatible wrapper
model_wrapper = PyTorchWrapper(model)

# Perform cross-validation
accuracy_scores = cross_val_score(model_wrapper, X_scaled, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(model_wrapper, X_scaled, y, cv=cv, scoring='f1')
precision = cross_val_score(model_wrapper, X_scaled, y, cv=cv, scoring='precision')
recall = cross_val_score(model_wrapper, X_scaled, y, cv=cv, scoring='recall')

# Print the cross-validation results
print("Accuracy scores:", accuracy_scores)
print("Mean accuracy:", accuracy_scores.mean())
print("F1 scores:", f1_scores)
print("Mean F1 score:", f1_scores.mean())
print("Mean Precision score:", precision.mean())
print("Mean Recall score:", recall.mean())

