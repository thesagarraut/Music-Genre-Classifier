import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# Load your trained model
model = load_model('E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/genre_classifier.h5')  # Use the correct path to your model

# Set up the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# Create the validation generator
val_generator = val_datagen.flow_from_directory(
    'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/dataset/split_images/val',  # Path to your validation data folder
    target_size=(128, 128),   # Resize to the size your model accepts
    batch_size=32,            # Number of images to process at once
    class_mode='categorical'  # Multi-class classification (one-hot encoded labels)
)

# Get the true labels from the generator
y_true = val_generator.classes  # The true labels from the validation generator

# Get the predicted labels from the model
y_pred = model.predict(val_generator)  # Model predictions on the validation set

# Convert predictions from probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class indices

# Get the class labels from the validation generator
class_labels = list(val_generator.class_indices.keys())  # List of genre names

# Call the function to plot the confusion matrix
plot_confusion_matrix(y_true, y_pred_classes, class_labels)
