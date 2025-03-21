# Private - we have User identifier

# Code without User Identifier (only API calls being made)

import json
import base64
import os
import pyttsx3
import nest_asyncio
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig

# Initialize Hive AI client
client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",  # Hive AI's endpoint
    api_key="HwDF5vDdbekdQbWsjcrsAXfsZo53N2v7"  # Replace with your API key
)

# Set up NeMo Guardrails
NVIDIA_API_KEY = "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"
nest_asyncio.apply()
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Initialize Text-to-Speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Adjust if needed
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 180)

def get_completion(prompt, image_path, model="meta-llama/llama-3.2-11b-vision-instruct"):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def nemo(text):
    completion = rails.generate(messages=[{"role": "user", "content": text}])
    return completion["content"]

output_file = "results.txt"

with open(output_file, "w") as result_file:
    image_name = "image.jpg"
    question = "Describe this scene"
    image_path = image_name # os.path.join(test_folder, image_name)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        final_response = "No Response because image not found."
    else:
        print(f"Processing {image_name}...")
        
        # Get AI response
        ai_response = get_completion(question, image_path)
        
        # Apply NeMo Guardrails
        final_response = ai_response
        # nemo(ai_response)
    
    # Write to output file
    result_file.write(f"Image: {image_name}\n")
    result_file.write(f"Question: {question}\n")
    result_file.write(f"Response: {final_response}\n\n")
    
    # Convert to speech
    engine.say(final_response)
    engine.runAndWait()
    
    print(f"Completed: {image_name}\n")
