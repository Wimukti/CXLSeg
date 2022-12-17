import tensorflow as tf

def parse_function(filename, text):
    # Read entire contents of image
    try:
        image_string = tf.io.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.io.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize image with padding to 244x244
        image = tf.image.resize_with_pad(image, 224, 224, method=tf.image.ResizeMethod.BILINEAR)

        return image, text
    except Exception as e:
        print(filename)
        print('Error', e)

# Data augmentation
def data_augmentation(image, text):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)

    # Random brightness, saturation and contrast shifting
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, text

# Convert image to grayscale
def convert_to_gray(image, text):
    image = tf.image.rgb_to_grayscale(image)
    return image, text