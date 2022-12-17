import tqdm
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from classification.utils import parse_function

# CheXpert classes
class_names = [
    'No Finding',
    'Enl. C. med.',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

# Path to generated test, train, and validate CSV folder
split_csv_root = 'path to generated CXLSeg split files'

# Path to CXLSeg dataset folder
CXLSeg_root = 'path to CXLSeg dataset folder'

# Set unsure values(-1) of CheXpert labeler to 0 or 1
replacements = {float('nan'): 0, -1.0: 1}

# Read test CSV file and set train image paths and labels
test_csv_file = os.path.join(split_csv_root, 'CXLSeg-validate.csv')
test_data = pd.read_csv(test_csv_file).replace(replacements).values
test_image_paths = [os.path.join(CXLSeg_root, path) for path in test_data[:, 0]]
test_labels = np.uint8(test_data[:, 1:])

# Create prefetch dataset from test data
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(16)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Load trained weights (Specify the filename of last saved weight)
model = tf.keras.models.load_model('./weights-xx.hdf5')
y_true_all = []
y_pred_all = []

for i, (x, y_true) in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
    y_true_all.append(y_true[0])
    y_pred = model.predict(x)
    y_pred_all.append(y_pred[0])


y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)


label_baseline_probs = []

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot

print('                     auc,     precision,   recall,      f_score,     sensitivity,    specificity,     accuracy')

for i in range(14):

    # Calculating AUC, Precision, Recall, F1 Score, Sensitivity, Specificity, and Accuracy for each class of CheXpert
    fpr, tpr, thresholds = roc_curve(y_true_all[:, i], y_pred_all[:, i])
    auc_score = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    acc_score = accuracy_score(y_true_all[:, i], y_pred_all[:, i] > optimal_threshold)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true_all[:, i], y_pred_all[:, i] > optimal_threshold, average='binary')
    specificity = 1 - fpr[optimal_idx]
    print( f'RES_{class_names[i]:<20} {auc_score:>.5f}, \t {precision:>.5f}, \t {recall:>.5f}, \t {f_score:>.5f}, \t {recall:>.5f}, \t {specificity:>.5f}, \t {acc_score:>.5f}')

    # Plotting ROC curves
    pyplot.title(class_names[i])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    fpr, tpr, thresholds = roc_curve(y_true_all[:, i], y_pred_all[:, i])
    auc_score = auc(fpr, tpr)
    pyplot.plot(fpr, tpr, marker=',', label="{}, AUC={:.3f}".format('ResNet50', auc_score))
    pyplot.legend(loc='lower right')

    pyplot.savefig(f'./roc_curve_{i}.png', bbox_inches='tight')
    pyplot.show()

