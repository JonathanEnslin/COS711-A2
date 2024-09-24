import pandas as pd

# Example list of lists representing the accuracies of different runs
runs = [
    [0.8, 0.82, 0.85],       # Run1
    [0.75, 0.78],            # Run2
    [0.82, 0.83, 0.86, 0.88],# Run3
    [0.79, 0.81],            # Run4
    [0.76],                  # Run5
    []
]

# Convert to DataFrame, padding with NaN to equalize the lengths
df = pd.DataFrame(runs).T  # Transpose so that each list becomes a column
df.columns = [f'Run{i+1}' for i in range(len(runs))]  # Rename columns to Run1, Run2, ..., Run5

# Display the DataFrame
print(df)
