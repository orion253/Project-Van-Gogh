import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load an image
img = plt.imread('/Users/linatlemcani/desktop/PVG/images/rainbow.png')

# Convert the image to grayscale
gray = np.mean(img, axis=2)

# Normalize the pixel values
gray_norm = gray / np.max(gray)

# Reshape the image into a 1D array
gray_flat = gray_norm.flatten()

# Define the sampling rate and duration of the audio
sr = 44100
length = len(gray_flat)

# Generate a sinusoidal tone for each pixel in the image
tones = [librosa.tone(np.abs(gray_flat[i]), sr=sr, length=length) for i in range(length)]

# Mix the tones together to create the final audio signal
audio = np.sum(tones, axis=0)

# Get the absolute path to the desktop folder
desktop = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

# Save the audio signal to a file on the desktop
audio_path = os.path.join(desktop, 'audio.wav')
librosa.output.write_wav(audio_path, audio, sr=sr)

# Display the original image and the spectrogram of the audio signal
plt.subplot(1, 2, 1)
plt.imshow(gray_norm, cmap='gray')
plt.title('Image')

plt.subplot(1, 2, 2)
spec = librosa.stft(audio)
librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=sr, x_axis='time', y_axis='log')
plt.title('Spectrogram')
print("finish!")

plt.show()
print("finish!")
