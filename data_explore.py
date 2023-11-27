import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data_heart.csv')
print(df.describe())

# Create histograms for each numerical feature
df.hist(figsize=(10, 10))
plt.tight_layout()
plt.show()

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

import numpy as np
from scipy.stats import spearmanr, kendalltau

# Assuming you have the heart failure dataset stored in 'data' variable

# Extract the columns of interest
age = df['DEATH_EVENT']  # Age
ejection_fraction = df['ejection_fraction']  # Ejection Fraction

# Calculate Spearman's rank correlation
spearman_corr, spearman_pvalue = spearmanr(age, ejection_fraction)
print("Spearman's correlation:", spearman_corr)
print("p-value:", spearman_pvalue)

# Calculate Kendall's rank correlation
kendall_corr, kendall_pvalue = kendalltau(age, ejection_fraction)
print("Kendall's correlation:", kendall_corr)
print("p-value:", kendall_pvalue)


