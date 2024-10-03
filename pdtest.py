import pandas as pd
import numpy as np

# Sample DataFrame (replace this with your actual DataFrame)
df = pd.DataFrame({
    'A': [1, 2, 3, np.nan, np.nan],
    # 'B': [np.nan, 4, 5, 6, np.nan],
    # 'C': [7, np.nan, np.nan, 8, 9],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaNs column
})

# Print the original DataFrame
print("Original DataFrame:\n", df)

# Find the last non-null value in each column, ignoring columns that are all NaN
last_non_null_values = df.apply(lambda col: col.dropna().iloc[-1] if col.dropna().size > 0 else np.nan)

# Filter out NaN values in case any column is entirely NaN
last_non_null_values = last_non_null_values.dropna()

# Print intermediate result: last non-null values in each column
print("\nLast non-null values in each column (excluding all-NaN columns):\n", last_non_null_values)

# Calculate the mean and standard deviation
mean_value = last_non_null_values.mean()
std_value = last_non_null_values.std()

# Print the final results
print("\nMean of final non-null elements:", mean_value)
print("Standard deviation of final non-null elements:", std_value)
