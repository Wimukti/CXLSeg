import pandas as pd
import os

# Path to CXLSeg CSV files
csv_root = 'metadata files'

# Set uncertain values of (-1) to 0 or 1
replacements = {float('nan'): 0, -1.0: 1}

# Read segmented and split files
segmented_csv_file = os.path.join(csv_root, 'CXLSeg-segmented.csv')
segmented = pd.read_csv(segmented_csv_file)
split_csv_file = os.path.join(csv_root, 'CXLSeg-split.csv')
split = pd.read_csv(split_csv_file)

# Merge both files to obtain a combined file of all the columns
merged = pd.merge(segmented, split)

# Obtain train, test, and validate data with DicomPath and CheXpert classes
train = merged.loc[merged['split'] == 'train']
train = train[train.columns[~train.columns.isin(['dicom_id', 'subject_id', 'study_id', 'split'])]]
test = merged.loc[merged['split'] == 'test']
test = test[test.columns[~test.columns.isin(['dicom_id', 'subject_id', 'study_id', 'split'])]]
validate = merged.loc[merged['split'] == 'validate']
validate = validate[validate.columns[~validate.columns.isin(['dicom_id', 'subject_id', 'study_id', 'split'])]]

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
