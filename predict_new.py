import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('genre_classifier.h5')

# Class names (same order as training)
class_names = ['blues', 'classical', 'hiphop', 'jazz', 'rock']  # Update if you used different ones

# Load and preprocess image
img_path = 'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/new_spectrograms/Mediterranean View - Everet Almond.png'  # e.g., 'test_spectrograms/rock_example.png'
img = image.load_img(img_path, target_size=(128, 128))  # Resize same as training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
print("Predicted Genre:", predicted_class)
