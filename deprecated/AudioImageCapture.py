import keyboard  # For detecting keypresses
import pyaudio   # For audio recording
import wave      # For saving the audio file

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sampling rate
CHUNK = 1024  # Size of each audio chunk
# RECORD_SECONDS = 5  # Duration of recording if fixed duration needed
WAVE_OUTPUT_FILENAME = "recording.wav"

audio = pyaudio.PyAudio()

# Function to record audio
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Press 'spacebar' again to stop.")
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed("space"):  # Stop recording when spacebar is pressed again
            break

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()

    # Save the recording
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {WAVE_OUTPUT_FILENAME}")

print("Press 'spacebar' to start recording...")
keyboard.wait("space")  # Wait for the first spacebar press
record_audio()

# Cleanup
audio.terminate()
