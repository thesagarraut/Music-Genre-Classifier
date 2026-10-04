#Training the Model Using ImageDataGenerator
#ImageDataGenerator to load the images, preprocess them, and start training your model.
#Resizing images

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import build_model  # Your CNN model code

# Directories for the split audio spectrogram images
train_dir = 'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/dataset/split_images/train'
val_dir = 'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/dataset/split_images/val'

# Parameters
batch_size = 32
img_height, img_width = 128, 128  # Resized image dimensions for model

# Image data generator (with augmentation for train data)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Image data generator for validation (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create train and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model (example CNN model)
model = build_model(input_shape=(img_height, img_width, 3), num_classes=train_generator.num_classes)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Save the trained model
model.save('genre_classifier.h5')
