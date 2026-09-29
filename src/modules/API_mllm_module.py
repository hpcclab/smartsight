import base64
import requests
import cv2
import json
from .ai_module_base import BaseAIModel
from .shared_buffer import video_buffer

class APIMLLMModule(BaseAIModel):
    def __init__(self):
        super().__init__("openrouter_api")
        self.api_key = self.config.get("api_key")
        # self.model_name = self.config.get("model", "gemma3:12b") # default model

    def load_model(self):
        # API module does not need to load local weights
        pass

    def run_inference(self, input_data: str, use_image: bool = False, model: str = None, stream: bool = False, frame=None, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_data}
                ]
            }
        ]

        if use_image:
            if frame is None:
                frame = video_buffer.retrieve_frame()
            if frame is not None:
                # Encode frame as JPEG base64
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            else:
                self.logger.warning("use_image is True, but no frame is available in the shared buffer.")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        response = None
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                def generate():
                    try:
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        data = json.loads(line[6:])
                                        if "choices" in data and len(data["choices"]) > 0:
                                            delta = data["choices"][0].get("delta", {})
                                            content = delta.get("content")
                                            if content:
                                                yield content
                                    except json.JSONDecodeError:
                                        pass
                    finally:
                        response.close()
                return generate()
            else:
                try:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                finally:
                    response.close()
        except Exception as e:
            self.logger.error(f"Error during API request: {e}")
            if response is not None and hasattr(response, 'text'):
                self.logger.error(f"Response details: {response.text}")
                response.close()
            if stream:
                def _empty():
                    if False:
                        yield ""
                return _empty()
            return None
