import pandas as pd

# Read the CSV files into pandas DataFrames
old = pd.read_csv('../inputs/_ata_training_pos_shuffle_neg_200.csv')
new = pd.read_csv('../inputs/ata_training_shuffle_neg_stride_200.csv')

# Initialize an empty DataFrame to store differences
differences = []

# Compare rows one by one
for idx in range(len(old)):
    if old.iloc[idx]['sequence'] != new.iloc[idx]['sequence']:
        differences.append(idx+1)

# Print the rows with differences
print(differences)
