import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance
from scipy.stats import ttest_ind



# Generate a 50x10 matrix of random values
matrix = np.random.randint(1, 11, size=(5, 2))

# Define the filename for the CSV file
file_name = "random_matrix.csv"

# Save the matrix to a CSV file
np.savetxt(file_name, matrix, delimiter=',')

# Load the matrix from the CSV file
loaded_matrix = np.loadtxt(file_name, delimiter=',')
print(loaded_matrix)

# Check for zeros and NaNs
has_zeros = np.any(loaded_matrix == 0)
has_nans = np.isnan(loaded_matrix).any()
print("zeros", has_zeros)
print("NaNs", has_nans)

# Value range for the entire matrix
min_value = np.min(loaded_matrix)
max_value = np.max(loaded_matrix)
print("value range", min_value, max_value)

# Value range for each row and each column
min_values_rows = np.min(loaded_matrix, axis=1)
max_values_rows = np.max(loaded_matrix, axis=1)
print("row value range", min_values_rows, max_values_rows)
min_values_columns = np.min(loaded_matrix, axis=0)
max_values_columns = np.max(loaded_matrix, axis=0)
print("column value range", min_values_columns, max_values_columns)

flat_vector = loaded_matrix.flatten()
print("flat vector", flat_vector)


first_group = loaded_matrix[:25, :]
second_group = loaded_matrix[25:, :]
print("first group", first_group)
print("second group", second_group)

even_rows = loaded_matrix[::2, :]
odd_rows = loaded_matrix[1::2, :]
print("even rows", even_rows)
print("odd rows", odd_rows)

transposed_matrix = loaded_matrix.T
print("transposed matrix", transposed_matrix)


# Column-wise Min-Max normalization
min_max_scaler = MinMaxScaler()
normalized_matrix_column = min_max_scaler.fit_transform(loaded_matrix)
print("normalized matrix", normalized_matrix_column)


# Row-wise Z-Score normalization
z_score_scaler = StandardScaler(with_mean=False)
normalized_matrix_row = z_score_scaler.fit_transform(loaded_matrix)
print("normalized matrix", normalized_matrix_row)


# Euclidean distance matrix
euclidean_distance_matrix = distance.cdist(loaded_matrix, loaded_matrix, 'euclidean')
print("euclidean distance matrix", euclidean_distance_matrix)

# Correlation distance matrix
correlation_distance_matrix = distance.cdist(loaded_matrix, loaded_matrix, 'correlation')
print("correlation distance matrix", correlation_distance_matrix)


even_means = np.mean(even_rows, axis=1)
odd_means = np.mean(odd_rows, axis=1)
t_stat, p_value = ttest_ind(even_means, odd_means)
print("t-statistic", t_stat, p_value)

if (p_value < 0.05):
    print("The difference between the means is statistically significant")
else:
    print("The difference between the means is not statistically significant")


