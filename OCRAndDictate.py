# Addtional requirements: pip install pytesseract pillow pyttsx3
import sys
from PIL import Image
import pytesseract
import pyttsx3

def extract_text(image_path):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    text = pytesseract.image_to_string(image)
    return text.strip()

#Might want to take this out if we are dictating text from other parts of the code 
def dictate_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    # Default to "output.jpg" if image not provided.
    image_path = sys.argv[1] if len(sys.argv) > 1 else "output.jpg"
    extracted_text = extract_text(image_path)
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
        dictate_text(extracted_text)
    else:
        print("No text detected in the image.")

if __name__ == "__main__":
    main() 


# To run this in other code, include: (with import subprocess)
# subprocess.run(["python", "smartsight/OCRAndDictate.py", "image_with_text.jpg"])
