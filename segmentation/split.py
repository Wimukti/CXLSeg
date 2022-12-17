import pandas as pd
import os

# Path to CXLSeg CSV files
csv_root = '../metadata files'

# Read segmented and split files
segmented_csv_file = os.path.join(csv_root, 'CXLSeg-segmented.csv')
segmented = pd.read_csv(segmented_csv_file)
split_csv_file = os.path.join(csv_root, 'CXLSeg-split.csv')
split = pd.read_csv(split_csv_file)
mask_csv_file = os.path.join(csv_root, 'CXLSeg-mask.csv')
mask = pd.read_csv(mask_csv_file)

# Rename the mask file column name before merging
mask = mask.rename(columns={'DicomPath': 'DicomPathMask'})

# Merge all files to obtain a combined file with all the columns
merged = pd.merge(segmented, split)
merged = pd.merge(merged, mask)

# Obtain train, test, and validate data with DicomPath and DicomPathMask
train = merged.loc[merged['split'] == 'train']
train = train[["DicomPath", "DicomPathMask"]]
test = merged.loc[merged['split'] == 'test']
test = test[["DicomPath", "DicomPathMask"]]
validate = merged.loc[merged['split'] == 'validate']
validate = validate[["DicomPath", "DicomPathMask"]]

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
