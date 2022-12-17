import pandas as pd
import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import Recall, Precision
from utils import *


# Global parameters
num_epochs = 20
batch_size = 5
lr = 1e-5

# Path to generated test, train, and validate CSV folder
# split_csv_root = 'path to generated CXLSeg split files'
split_csv_root = './'

# Path to CXLSeg dataset folder
# CXLSeg_root = 'path to CXLSeg dataset folder'
CXLSeg_root = 'E:/mimic-cxr-jpg-2.0.0.physionet.org-cxeseg'

# Path to MIMIC-CXR-JPG dataset folder
# MIMIC_root = 'path to CXLSeg dataset folder'
MIMIC_root = 'E:/mimic-cxr-jpg-2.0.0.physionet.org'

# Read train CSV file and set train image and mask paths
train_csv_file = os.path.join(split_csv_root, 'CXLSeg-train.csv')
train_data = pd.read_csv(train_csv_file).values
train_image_paths = [os.path.join(MIMIC_root, path) for path in train_data[:, 0]]
train_mask_paths = [os.path.join(CXLSeg_root, path) for path in train_data[:, 1]]

# Read validate CSV file and set validate image and mask paths
validate_csv_file = os.path.join(split_csv_root, 'CXLSeg-validate.csv')
validate_data = pd.read_csv(validate_csv_file).values
validate_image_paths = [os.path.join(MIMIC_root, path) for path in validate_data[:, 0]]
validate_mask_paths = [os.path.join(CXLSeg_root, path) for path in validate_data[:, 1]]

train_dataset = tf_dataset(train_image_paths, train_mask_paths, batch=batch_size)
validate_dataset = tf_dataset(validate_image_paths, validate_mask_paths, batch=batch_size)

# Unet Model
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

# Define model
input_shape = (H, W, 3)
model = build_unet(input_shape)
metrics = [dice_coef, iou, Recall(), Precision()]
model.compile(loss=bce_dice_loss, optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics)

callbacks = [
    ModelCheckpoint(r'./model.hdf5', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
]

model_history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=validate_dataset,
    callbacks=callbacks
)

# Save model and history
model.save(r'./model.hdf5')
np.save(r'./model-history.npy', model_history.history)