import os
import time
import json
import base64
import wave
import pyaudio
import whisper
import pyttsx3
import nest_asyncio
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig
from operations.commands import Commands
from paddleocr import PaddleOCR
from operations.commands import build_ocr, Commands

class ActiveMode:
    def __init__(self):
        self.whisperModel = whisper.load_model("base")
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('rate', 150)

        # Set up NeMo Guardrails
        NVIDIA_API_KEY = "key1"
        nest_asyncio.apply()
        os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
        config = RailsConfig.from_path("./config")
        self.rails = LLMRails(config)

        # Initialize Hive AI client
        self.client = OpenAI(
            base_url="https://api.thehive.ai/api/v3/",
            api_key="key2"
        )
        # Initialize command handler for OCR commands
        ocr_engine = build_ocr()
        self.commands = Commands(ocr_engine, conf_threshold=0.65, min_length=2)

    def record_audio(self, audioObj):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        WAVE_OUTPUT_FILENAME = "recording.wav"

        stream = audioObj.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        print("Recording... release spacebar to stop.")
        frames = []

        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if not __import__('keyboard').is_pressed("space"): 
                break
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()

        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audioObj.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved as {WAVE_OUTPUT_FILENAME}")

    def recognize_speech(self):
        transcription = self.whisperModel.transcribe("recording.wav")
        print(transcription["text"])
        return transcription["text"]

    def get_completion(self, prompt, image_path, model="meta-llama/llama-3.2-11b-vision-instruct"):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]}
            ],
            temperature=0.7,
            max_tokens=35
        )
        return response.choices[0].message.content

    def nemo(self, text):
        completion = self.rails.generate(messages=[{"role": "user", "content": text}])
        return completion["content"]

    def MLLMAnalyzeImage(self, UserRequest, UploadImage_path, output_file="results.txt"):
        # Check for 'command' mode: OCR text reading
        req_lower = "read this text." #UserRequest.lower()
        if ("read" in req_lower or "text" in req_lower):
            lines = self.commands.read_text(UploadImage_path)
            if lines:
                paragraph = ". ".join(lines)
                print("Command read text:", paragraph)
                self.engine.say(paragraph)
                self.engine.runAndWait()
            elif not lines or len(lines) <= 0:
                print("No text detected")
                self.engine.say("No text detected")
            self.engine.runAndWait()
            return
        with open(output_file, "a") as result_file:
            image_name = UploadImage_path
            prompt = UserRequest + " Use the picture to appropriately answer the prompt. Ensure response is reasonable, brief, and accurate to the image. Do your best to answer regardless of grammar issues."
            image_path = image_name

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                final_response = "No Response because image not found."
            else:
                print(f"Processing {image_name}...")
                ai_response = self.get_completion(prompt, image_path)
                final_response = self.nemo(ai_response)

            self.engine.say(final_response)
            self.engine.runAndWait()

            result_file.write(f"Image: {image_name}\n")
            result_file.write(f"Prompt: {prompt}\n")
            result_file.write(f"Original Response: {ai_response}\n")
            result_file.write(f"Nemo Guardrails: {final_response}\n\n")
            result_file.write("---------------------\n") 
