from ollama import chat
from .ai_module_base import BaseAIModel

class EdgeMLLMModule(BaseAIModel):
    """
    Module responsible for generating text and interacting with local Ollama models.
    Replaces the functionality of junk/ModelManager.py.
    """
    def __init__(self):
        super().__init__("ollama_llm")
        self.model_name = self.config.get("model", "gemma3:4b")

    def load_model(self):
        """Loads the local Ollama LLM into memory."""
        if not self.config.get("ollama_llm_enabled", "gemma3:4b"):
            self.logger.warning("Ollama LLM is not enabled in the configuration.")
            return False
        self.logger.info(f"Loading Edge MLLM model ({self.model_name})...")
        try:
            # Trigger a small load to ensure model is in VRAM if needed
            chat(model=self.model_name, messages=[{'role':'user','content':'ping'}])
            self.model = self.model_name # Assign simply to denote loaded state
            self.logger.info("Local LLM Loaded successfully.")
        except Exception as e:
            self.logger.error(f"Could not load Local LLM. Ensure Ollama is running. Error: {e}")
            raise

    def _image_payload(self, image_path, frame, use_image):
        """Return an Ollama images list, or None when this call is text-only."""
        if image_path:
            return [image_path]
        if not use_image and frame is None:
            return None
        if frame is None:
            from .shared_buffer import video_buffer
            frame = video_buffer.retrieve_frame()
        if frame is None:
            self.logger.warning("use_image is True, but no frame is available in the shared buffer.")
            return None
        import cv2
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            self.logger.warning("Could not encode the frame for Ollama.")
            return None
        return [buffer.tobytes()]

    def run_inference(self, input_data: str, **kwargs):
        """
        Generator function that streams text from Ollama.

        Args:
            input_data (str): The prompt text to send to the model.
            kwargs:
                image_path (str): Optional file path for vision generation.
                frame: Optional image array. JPEG-encoded and sent to Ollama.
                use_image (bool): When True and frame is omitted, use the shared camera frame.
                already_spoken (str): Optional assistant context when no image is attached.

        Yields:
            str: Chunks of the generated response.
        """
        prompt = input_data
        image_path = kwargs.get("image_path")
        frame = kwargs.get("frame")
        use_image = kwargs.get("use_image", False)
        already_spoken = kwargs.get("already_spoken")

        try:
            images = self._image_payload(image_path, frame, use_image)
            if images is not None:
                complete_prompt = f"Concisely answer this query in paragraph form using the image. {prompt}"
                messages = [
                    {
                        'role': 'user',
                        'content': complete_prompt,
                        'images': images
                    }
                ]
            elif already_spoken is not None:
                messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': already_spoken}
                ]
            else:
                messages = [
                    {'role': 'user', 'content': prompt}
                ]

            stream = chat(
                model=self.model_name,
                messages=messages,
                stream=True
            )

            try:
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        content = content.replace("*", "")
                        yield content
            finally:
                closer = getattr(stream, "close", None)
                if callable(closer):
                    closer()

        except Exception as e:
            self.logger.error(f"Ollama Error: {e}")
            yield f" [Local Error: {str(e)}] "
