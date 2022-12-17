import pandas as pd
from utils import *
import os
import numpy as np
import tensorflow as tf

# Global parameters
batch_size = 5

# Path to generated test, train, and validate CSV folder
split_csv_root = 'path to generated CXLSeg split files'

# Path to CXLSeg dataset folder
CXLSeg_root = 'path to CXLSeg dataset folder'

# Path to MIMIC-CXR-JPG dataset folder
MIMIC_root = 'path to CXLSeg dataset folder'

# Read validate CSV file and set test image and mask paths
test_csv_file = os.path.join(split_csv_root, 'CXLSeg-test.csv')
test_data = pd.read_csv(test_csv_file).values
test_image_paths = [os.path.join(MIMIC_root, path) for path in test_data[:, 0]]
test_mask_paths = [os.path.join(CXLSeg_root, path) for path in test_data[:, 1]]

# Create prefetch dataset of test data
test_dataset = tf_dataset(test_image_paths, test_mask_paths, batch=batch_size)

# Load Model and model history
model = tf.keras.models.load_model(r'\model.hdf5', custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef, "iou": iou})
model_history = np.load(r'model-history.npy', allow_pickle=True)

loss = model_history.item().get('loss')
val_loss = model_history.item().get('val_loss')
acc = model_history.item().get('iou')
val_acc = model_history.item().get('val_iou')

# Plot training and validation graphs for accuracy and loss
fig, ax = plt.subplots()
epochs = range(1, len(loss) + 1)
ax.plot(epochs, acc, marker='.', label='Training Accuracy')
ax.plot(epochs, val_acc, marker='.', label='Validation Accuracy')
ax.plot(epochs, loss, marker='.', label='Training loss')
ax.plot(epochs, val_loss, marker='.', label='Validation loss')

# ax.title('Training and validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy/Loss')
plt.legend()
plt.savefig(f'./training_validation_accuracy_loss.png', bbox_inches='tight')
plt.show()

print("Evaluate metrics")
result = model.evaluate(test_dataset)
result = dict(zip(model.metrics_names, result))
print(result)