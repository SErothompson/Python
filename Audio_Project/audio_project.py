import sounddevice as sd
import numpy as np
import speech_recognition as sr
import tkinter as tk

# Define recording parameters
samplerate = 44100  # Sample rate in Hertz
duration = 5  # Duration of recording in seconds
channels = 1  # Number of audio channels

print("Recording...")

# Record audio
recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int16')
sd.wait()  # Wait until recording is finished

print("Recording complete")

# Convert the recorded audio to a suitable format for speech recognition
audio_data = np.array(recording).flatten()

# Use the recognizer
recognizer = sr.Recognizer()

# Convert the numpy array to audio data
audio = sr.AudioData(audio_data.tobytes(), samplerate, 2)

# Initialize the text variable
text = ""

try:
    # Use Google's speech recognition to convert audio to text
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio.")
except sr.RequestError as e:
    print("Error: " + str(e))

# Display the recognized text in a GUI
root = tk.Tk()
root.title("Audio Recognition")
root.geometry("300x200")

label = tk.Label(root, text=text)
label.pack()

root.mainloop()
