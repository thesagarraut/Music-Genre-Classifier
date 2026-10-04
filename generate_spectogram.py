import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# Input and output paths
input_folder = 'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/new_audio'
output_folder = 'E:/College Mtech/Sem 2/AMLDL Lab/8 Mini Project/Mini Project/new_spectrograms'

# Create output folder if not exist
os.makedirs(output_folder, exist_ok=True)

# Loop through audio files
for filename in os.listdir(input_folder):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        file_path = os.path.join(input_folder, filename)

        # Load audio file
        y, sr = librosa.load(file_path, duration=30)

        # Create Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save spectrogram
        plt.figure(figsize=(2.56, 2.56), dpi=50)  # 128x128 pixels
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')  # Remove axes
        out_name = os.path.splitext(filename)[0] + '.png'
        plt.savefig(os.path.join(output_folder, out_name), bbox_inches='tight', pad_inches=0)
        plt.close()

print("✅ Spectrograms saved in", output_folder)
