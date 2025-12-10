import os
import time
import json
import base64
import wave
import pyaudio
import whisper
import pyttsx3
import nest_asyncio
from cv2 import imwrite
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig
from operations.commands import Commands
from paddleocr import PaddleOCR
from pathlib import Path
import base64
import requests
from io import BytesIO

class ActiveMode:
    def __init__(self, ocr_engine):
        self.whisperModel = whisper.load_model("base")
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('rate', 150)

        # Set up NeMo Guardrails
        KeyFile = Path(__file__).parent.parent.parent / "configSensitive" / "apikeys.txt"
        with open(KeyFile, "r") as file:
            NVIDIA_API_KEY = file.readline().strip()
            HIVE_API_KEY = file.readline().strip()
        nest_asyncio.apply()
        os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
        config = RailsConfig.from_path("./config")
        self.rails = LLMRails(config)

        # Initialize Hive AI client
        self.client = OpenAI(
            base_url="https://api.thehive.ai/api/v3/",
            api_key=HIVE_API_KEY
        )
        # Initialize command handler for OCR commands
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
    
    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_completion2(self, prompt, image_path, model="google/gemma-3-4b-it:free"):
        OPENROUTER_API_KEY = 'PLACEHOLDER'
        ########### NEW STUFF
        try:
            api_key = OPENROUTER_API_KEY

            if not prompt:
                print('error: No prompt provided.')
                return "error: No prompt provided."
            base64_image = self.encode_image_to_base64(image_path)
            data_url = f"data:image/jpeg;base64,{base64_image}"
                
            # Construct the vision model payload
            # The 'content' is now a list of parts (text and image)

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ]

            payload = {
                "model": model,
                "messages": messages
            }
            response = requests.post(url, headers=headers, json=payload)


            # Handle potential errors from OpenRouter
            response.raise_for_status() # Raises an exception for bad status codes

            # Parse the JSON response
            result = response.json()
            
            # Handle cases where the response might be empty or malformed
            if not result.get('choices') or not result['choices'][0].get('message'):
                print('Invalid response from model.')
                return 'Invalid response from model.'
                
            model_response = result['choices'][0]['message']['content']
            return model_response
            
        except requests.exceptions.RequestException as e:
            # Provide more detail on API errors
            error_details = e.response.text if e.response else str(e)
            print(f'API request failed: {error_details}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
        ########### END NEW STUFF

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

    def MLLMAnalyzeImage(self, UserRequest, img, output_file="MLLM-Results.txt"):
        # Check for 'command' mode: OCR text reading
        req_lower = UserRequest.lower()
        if ("read" in req_lower or "text" in req_lower):
            lines = self.commands.read_text(img)
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
            prompt = UserRequest + " Use the picture to appropriately answer the prompt. Ensure response is reasonable, brief, and accurate to the image. Do your best to answer regardless of grammar issues."

            print(f"Processing image...")
            imgName = "MLLMImg.jpg"
            imwrite(imgName, img)
            ai_response = self.get_completion2(prompt, imgName)

            
            # final_response = self.nemo(ai_response)
            final_response = ai_response # TEMP: For testing.


            self.engine.say(final_response)
            self.engine.runAndWait()

            result_file.write(f"Prompt: {prompt}\n")
            result_file.write(f"Original Response: {ai_response}\n")
            result_file.write(f"Nemo Guardrails: {final_response}\n\n")
            result_file.write("---------------------\n") 
            
