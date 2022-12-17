import pandas as pd
import os

# Path to CXLSeg CSV files
csv_root = '../metadata files'

# Read segmented and split files
segmented_csv_file = os.path.join(csv_root, 'CXLSeg-segmented.csv')
segmented = pd.read_csv(segmented_csv_file)
split_csv_file = os.path.join(csv_root, 'CXLSeg-split.csv')
split = pd.read_csv(split_csv_file)
metadata_csv_file = os.path.join(csv_root, 'CXLSeg-metadata.csv')
metadata = pd.read_csv(metadata_csv_file)

# Merge all files to obtain a combined file with all the columns
merged = pd.merge(segmented, split)
merged = pd.merge(merged, metadata)

# Obtain train, test, and validate data with DicomPath and DicomPathMask
train = merged.loc[merged['split'] == 'train']
train = train[["DicomPath", "Reports"]]
test = merged.loc[merged['split'] == 'test']
test = test[["DicomPath", "Reports"]]
validate = merged.loc[merged['split'] == 'validate']
validate = validate[["DicomPath", "Reports"]]

# Saving test, train and validate to separate csv files
train.to_csv(f'CXLSeg-train.csv', index=False)
test.to_csv(f'CXLSeg-test.csv', index=False)
validate.to_csv(f'CXLSeg-validate.csv', index=False)

print(f"Train: {train.shape[0]}")
print(f"Test: {test.shape[0]}")
print(f"Validate: {validate.shape[0]}")

'''
(194660, 15)
(24332, 15)
(24332, 15)
'''
