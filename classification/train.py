import pandas as pd
from utils import *
import os
import numpy as np
import tensorflow as tf

# Path to generated test, train, and validate CSV folder
split_csv_root = 'path to generated CXLSeg split files'

# Path to CXLSeg dataset folder
CXLSeg_root = 'path to CXLSeg dataset folder'

# Set unsure values(-1) of CheXpert labeler to 0 or 1
replacements = {float('nan'): 0, -1.0: 1}

# Read train CSV file and set train image paths and labels
train_csv_file = os.path.join(split_csv_root, '../CXLSeg-train.csv')
train_data = pd.read_csv(train_csv_file).replace(replacements).values
train_image_paths = [os.path.join(CXLSeg_root, path) for path in train_data[:, 0]]
train_labels = np.uint8(train_data[:, 1:])

# Read validate CSV file and set test image paths and labels
validate_csv_file = os.path.join(split_csv_root, '../CXLSeg-validate.csv')
validate_data = pd.read_csv(validate_csv_file).replace(replacements).values
validate_image_paths = [os.path.join(CXLSeg_root, path) for path in validate_csv_file[:, 0]]
validate_labels = np.uint8(validate_csv_file[:, 1:])

# Create prefetch dataset from train data
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(convert_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Create prefetch dataset from validate data
valid_dataset = tf.data.Dataset.from_tensor_slices((validate_image_paths, validate_labels))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.map(convert_to_gray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(16)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Define model (We have used ResNet50 in this case)
model = tf.keras.Sequential([
    tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 1)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(14, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='./weights-{epoch:02d}-{val_accuracy:.2f}.hdf5',
    monitor='val_loss', verbose=1)


class cnn_checkpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Sanity check: {self.model.layers[0].name}')
        self.model.layers[0].save_weights(f'./model_{epoch}.hdf5')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

callback = [model_checkpoint, cnn_checkpoint(), early_stopping]

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=10,
          callbacks=callback)
